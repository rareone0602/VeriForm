import Lean
import Mathlib

-- Theorem negation via Lean 4 metaprogramming.
-- Replaces the regex-based get_refute_theorem from the Python prover.
-- Built as `lake exe negate`. Driver protocol: see `main` at the bottom.
--
-- Mathlib is imported so that the parser knows about all the notation used
-- in formalised theorems (Type*, ∀, →, ∈, instance binders, etc).

open Lean Parser

namespace VeriForm.Negate

inductive Mode where
  | strong
  | full
  deriving Inhabited, Repr

def Mode.ofString? : String → Option Mode
  | "strong" => some .strong
  | "full"   => some .full
  | _        => none

instance : ToString Mode where
  toString
    | .strong => "strong"
    | .full   => "full"

/-- Slice the original source between two `String.Pos` positions. -/
private def sliceSource (src : String) (start : String.Pos) (stop : String.Pos) : String :=
  src.extract start stop

/-- Find the first descendant of `stx` whose `SyntaxNodeKind` matches `kind`. -/
private partial def findFirst? (stx : Syntax) (kind : SyntaxNodeKind) : Option Syntax :=
  if stx.getKind == kind then some stx
  else
    match stx with
    | .node _ _ args => args.findSome? (findFirst? · kind)
    | _ => none

/-- Extract `(theoremNameSlice, binderSourceSlice, typeSourceSlice)` from a parsed `command`
    syntax that wraps a `theorem` declaration. Returns substrings of the original `src`. -/
private def extractParts (src : String) (cmd : Syntax) : Except String (String × String × String) := do
  -- The expected shape (Lean 4.9):
  --   command/declaration/(modifiers, theorem-decl)
  --   where theorem-decl =
  --     node `Lean.Parser.Command.theorem #[
  --       atom "theorem",
  --       declId (ident with optional universe params),
  --       declSig (bracketedBinder* (":" term)?),  -- for theorem: `:` is required
  --       declVal
  --     ]
  let some thmStx := findFirst? cmd ``Lean.Parser.Command.theorem
    | .error "input does not contain a `theorem` declaration"
  let args := thmStx.getArgs
  -- args = [theorem-keyword, declId, declSig, declVal]
  if args.size < 4 then
    .error s!"unexpected theorem arity: {args.size}"
  let declId  := args[1]!
  let declSig := args[2]!
  -- declSig structure: node `Lean.Parser.Command.declSig #[binders, ":" + term]
  -- where binders is a `null` node containing zero or more bracketedBinders.
  let sigArgs := declSig.getArgs
  if sigArgs.size < 2 then
    .error s!"unexpected declSig arity: {sigArgs.size}"
  let bindersStx := sigArgs[0]!
  let typeColonStx := sigArgs[1]!  -- node containing ":" then the term
  let typeArgs := typeColonStx.getArgs
  if typeArgs.size < 2 then
    .error s!"unexpected type-colon arity: {typeArgs.size}"
  let typeTerm := typeArgs[1]!

  -- Extract the ident text from declId (declId itself may carry universe params; we
  -- strip those — the negated theorem doesn't need universe params duplicated
  -- because we just refer to the original by name in the negation body).
  let ident :=
    match declId.getArgs.get? 0 with
    | some idStx => idStx
    | none => declId
  let nameStr := ident.getId.toString

  -- Slice binders and type from source by their syntactic ranges.
  let binderStr ← match bindersStx.getRange? (canonicalOnly := true) with
    | some ⟨b, e⟩ => .ok (sliceSource src b e).trim
    | none        => .ok ""  -- empty binder list has no range
  let typeStr ← match typeTerm.getRange? (canonicalOnly := true) with
    | some ⟨b, e⟩ => .ok (sliceSource src b e).trim
    | none        => .error "could not locate theorem type in source"
  return (nameStr, binderStr, typeStr)

/-- Build the negated theorem string. Caller supplies an `Environment` whose
syntax categories include `command` (any env from `importModules #[`Init]`
suffices). -/
def negateTheorem (env : Environment) (mode : Mode) (src : String) : Except String String := do
  let cmd ← match Parser.runParserCategory env `command src "<input>" with
    | .ok s => .ok s
    | .error e => .error s!"parse error: {e}"
  let (name, binders, type) ← extractParts src cmd
  -- Match the naming used by the original regex implementation so existing
  -- downstream code (prompt construction, theorem-extractor matching) sees
  -- the same theorem identifier shape.
  let newName :=
    match mode with
    | .strong => s!"not_{name}_strong"
    | .full   => s!"not_{name}_full"
  let body :=
    match mode with
    | .strong =>
      if binders.isEmpty then
        s!"theorem {newName} : ¬ ({type}) := by sorry"
      else
        s!"theorem {newName} {binders} : ¬ ({type}) := by sorry"
    | .full =>
      if binders.isEmpty then
        s!"theorem {newName} : ¬ ({type}) := by sorry"
      else
        s!"theorem {newName} : ¬ (∀ {binders}, {type}) := by sorry"
  return body

end VeriForm.Negate

/-! ## Executable driver.

Line-protocol: each request is one line `<mode>\t<base64-src>` (base64 so the
source can contain newlines/tabs without escaping). Each response is one line
`OK\t<base64-result>` or `ERR\t<base64-message>`. EOF on stdin → exit 0.

Base64 chosen over JSON because: (a) zero parsing overhead, (b) no risk of
breaking on unusual unicode in Lean source, (c) framing is one line each way. -/

open VeriForm.Negate

private def b64decode (s : String) : Except String String :=
  match (Lean.Json.parse s!"\"{s}\"") with
  | _ => -- Json doesn't help us here; use ByteArray helpers from core
    .ok ""  -- placeholder, real impl below

namespace VeriForm.Negate.Driver

/-- Minimal base64 decode. We avoid pulling in Mathlib so we ship our own.  -/
private def base64Alphabet : String :=
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

private def base64CharIndex (c : Char) : Option Nat :=
  base64Alphabet.find (· == c) |>.byteIdx |> some |>.bind fun i =>
    if i < base64Alphabet.utf8ByteSize then some i else none

private def b64decodeBytes (s : String) : Except String ByteArray := Id.run do
  let mut out : ByteArray := .empty
  let mut buf : UInt32 := 0
  let mut bits : Nat := 0
  for c in s.toList do
    if c == '=' then break
    if c == '\n' || c == '\r' || c == ' ' || c == '\t' then continue
    match base64CharIndex c with
    | none => return .error s!"invalid base64 char: {c}"
    | some idx =>
      buf := (buf <<< 6) ||| (UInt32.ofNat idx)
      bits := bits + 6
      if bits >= 8 then
        bits := bits - 8
        let byte := ((buf >>> (UInt32.ofNat bits)) &&& 0xFF).toUInt8
        out := out.push byte
  return .ok out

private def b64encode (bs : ByteArray) : String := Id.run do
  let alphabet := base64Alphabet.toList.toArray
  let mut out : String := ""
  let n := bs.size
  let mut i := 0
  while i + 2 < n do
    let b0 := bs[i]!.toNat
    let b1 := bs[i+1]!.toNat
    let b2 := bs[i+2]!.toNat
    let triple := (b0 <<< 16) ||| (b1 <<< 8) ||| b2
    out := out.push alphabet[(triple >>> 18) &&& 0x3F]!
    out := out.push alphabet[(triple >>> 12) &&& 0x3F]!
    out := out.push alphabet[(triple >>> 6)  &&& 0x3F]!
    out := out.push alphabet[ triple         &&& 0x3F]!
    i := i + 3
  if i < n then
    let b0 := bs[i]!.toNat
    let b1 := if i+1 < n then bs[i+1]!.toNat else 0
    let triple := (b0 <<< 16) ||| (b1 <<< 8)
    out := out.push alphabet[(triple >>> 18) &&& 0x3F]!
    out := out.push alphabet[(triple >>> 12) &&& 0x3F]!
    if i+1 < n then
      out := out.push alphabet[(triple >>> 6) &&& 0x3F]!
      out := out.push '='
    else
      out := out.push '='
      out := out.push '='
  return out

def base64DecodeStr (s : String) : Except String String := do
  let bs ← b64decodeBytes s
  return String.fromUTF8! bs

def base64EncodeStr (s : String) : String :=
  b64encode s.toUTF8

def handleLine (env : Environment) (line : String) : String :=
  let parts := line.splitOn "\t"
  match parts with
  | [modeStr, b64src] =>
    match Mode.ofString? modeStr with
    | none => "ERR\t" ++ base64EncodeStr s!"unknown mode: {modeStr}"
    | some mode =>
      match base64DecodeStr b64src with
      | .error e => "ERR\t" ++ base64EncodeStr s!"base64 decode failed: {e}"
      | .ok src =>
        match negateTheorem env mode src with
        | .error e => "ERR\t" ++ base64EncodeStr e
        | .ok out  => "OK\t"  ++ base64EncodeStr out
  | _ => "ERR\t" ++ base64EncodeStr s!"expected 2 tab-separated fields, got {parts.length}"

end VeriForm.Negate.Driver

open VeriForm.Negate.Driver

def main : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[`Init, `Mathlib] {} (trustLevel := 1024)
  let stdin  ← IO.getStdin
  let stdout ← IO.getStdout
  -- Signal readiness so the Python wrapper knows we're up.
  stdout.putStrLn "READY"
  stdout.flush
  while true do
    let line ← stdin.getLine
    if line.isEmpty then break  -- EOF
    let trimmed := line.trim
    if trimmed.isEmpty then continue
    stdout.putStrLn (handleLine env trimmed)
    stdout.flush
