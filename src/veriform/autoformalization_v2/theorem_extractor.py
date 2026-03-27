import re
from typing import Optional, Tuple, List

class TheoremExtractor:
    def __init__(self):
        pass

    def _generate_mask(self, code: str) -> List[bool]:
        """
        Generates a boolean mask: True for active code, False for comments/strings.
        """
        n = len(code)
        mask = [True] * n
        i = 0
        in_string = False
        in_comment_line = False
        comment_block_depth = 0

        while i < n:
            # 1. String Content
            if in_string:
                mask[i] = False
                if code[i] == '"' and (i == 0 or code[i-1] != '\\'):
                    in_string = False
                i += 1
                continue
            
            # 2. Block Comment Content
            if comment_block_depth > 0:
                mask[i] = False
                if i + 1 < n:
                    if code[i:i+2] == '/-':
                        comment_block_depth += 1
                        mask[i+1] = False
                        i += 2
                        continue
                    elif code[i:i+2] == '-/':
                        comment_block_depth -= 1
                        mask[i+1] = False
                        i += 2
                        continue
                i += 1
                continue

            # 3. Line Comment Content
            if in_comment_line:
                if code[i] == '\n':
                    in_comment_line = False
                else:
                    mask[i] = False
                i += 1
                continue

            # 4. Start of String
            if code[i] == '"':
                in_string = True
                mask[i] = False
                i += 1
                continue

            # 5. Start of Block Comment
            if i + 1 < n and code[i:i+2] == '/-':
                comment_block_depth = 1
                mask[i] = False
                mask[i+1] = False
                i += 2
                continue
            
            # 6. Start of Line Comment
            if i + 1 < n and code[i:i+2] == '--':
                in_comment_line = True
                mask[i] = False
                mask[i+1] = False
                i += 2
                continue
            
            i += 1
        return mask

    def _remove_comments(self, code: str) -> str:
        """
        Returns a canonical version of the code with comments removed.
        Block comments replaced by space. Line comments removed.
        """
        n = len(code)
        result = []
        i = 0
        in_string = False
        in_comment_line = False
        comment_block_depth = 0
        
        while i < n:
            if in_string:
                result.append(code[i])
                if code[i] == '"' and (i == 0 or code[i-1] != '\\'):
                    in_string = False
                i += 1
                continue
            
            if comment_block_depth > 0:
                if i + 1 < n:
                    if code[i:i+2] == '/-':
                        comment_block_depth += 1
                        i += 2
                        continue
                    elif code[i:i+2] == '-/':
                        comment_block_depth -= 1
                        if comment_block_depth == 0:
                             result.append(" ") 
                        i += 2
                        continue
                i += 1
                continue

            if in_comment_line:
                if code[i] == '\n':
                    in_comment_line = False
                    result.append('\n')
                i += 1
                continue

            if code[i] == '"':
                in_string = True
                result.append('"')
                i += 1
                continue

            if i + 1 < n and code[i:i+2] == '/-':
                comment_block_depth = 1
                i += 2
                continue
            
            if i + 1 < n and code[i:i+2] == '--':
                in_comment_line = True
                i += 2
                continue
            
            result.append(code[i])
            i += 1
            
        return "".join(result)

    def get_last_theorem(self, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        mask = self._generate_mask(code)
        
        # 1. Find LAST "theorem"
        theorem_indices = []
        for match in re.finditer(r"\btheorem\b", code):
            if mask[match.start()]:
                theorem_indices.append(match.start())
        
        if not theorem_indices:
            return None, None, None

        start_idx = theorem_indices[-1]
        
        # 2. Extract Name
        name_match = re.match(r"theorem\s+(?P<name>[\w'.]+)", code[start_idx:])
        if not name_match:
            return None, None, None
        
        name = name_match.group("name")
        scan_idx = start_idx + name_match.end()
        
        # 3. Scan for Separators
        param_split_index = -1
        proof_split_index = -1
        
        bracket_balance = 0
        let_depth = 0
        n = len(code)
        i = scan_idx
        
        while i < n:
            if not mask[i]:
                i += 1
                continue
                
            char = code[i]
            
            # Brackets
            if char in "([{":
                bracket_balance += 1
            elif char in ")]}":
                bracket_balance -= 1
                if bracket_balance < 0: bracket_balance = 0
            
            if bracket_balance == 0:
                # Check for Proof Separator :=
                # (Must check this before simple colon)
                if char == ':' and i + 1 < n and code[i+1] == '=':
                    if let_depth > 0:
                        let_depth -= 1
                        i += 1 
                    else:
                        proof_split_index = i
                        break
                
                # Check for Type Separator :
                elif char == ':':
                    if param_split_index == -1:
                        param_split_index = i
                    else:
                        # HEURISTIC FOR DOUBLE COLON (theorem t : (params) : body)
                        # If we already found a colon, check if the text BEFORE it was effectively empty.
                        # If so, and the text BETWEEN the two colons looks like params (starts with bracket),
                        # treat the first colon as a prefix/spurious.
                        
                        pre_colon_segment = code[scan_idx : param_split_index]
                        inter_colon_segment = code[param_split_index+1 : i]
                        
                        cleaned_pre = self._remove_comments(pre_colon_segment).strip()
                        cleaned_inter = self._remove_comments(inter_colon_segment).strip()
                        
                        if not cleaned_pre and cleaned_inter and cleaned_inter[0] in "([{":
                            # Shift scan start to skip the first colon
                            scan_idx = param_split_index + 1
                            # Update split index to current colon
                            param_split_index = i

                # Check for let/have
                elif char == 'l' or char == 'h':
                    is_start = (i == 0 or not code[i-1].isalnum() and code[i-1] != '_')
                    if is_start:
                        if code[i:i+3] == 'let' and (i+3 >= n or not code[i+3].isalnum() and code[i+3] != '_'):
                            let_depth += 1
                        elif code[i:i+4] == 'have' and (i+4 >= n or not code[i+4].isalnum() and code[i+4] != '_'):
                            let_depth += 1

            i += 1
            
        if proof_split_index == -1:
            return name, None, None

        # 4. Extract
        if param_split_index != -1:
            params_raw = code[scan_idx:param_split_index]
            body_raw = code[param_split_index+1:proof_split_index]
            
            params_clean = self._remove_comments(params_raw).strip()
            body_clean = self._remove_comments(body_raw).strip()
            
            return name, (params_clean if params_clean else None), body_clean
        else:
            # No colon found: everything before := is params (binders)
            params_raw = code[scan_idx:proof_split_index]
            params_clean = self._remove_comments(params_raw).strip()
            # Body is None because there was no explicit type signature
            return name, (params_clean if params_clean else None), None