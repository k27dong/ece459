import os, re, sys
from typing import List

class FormatError(Exception):
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return f"Format error: {self.msg}"

class EndOfFileError(FormatError):
    def __init__(self, expected_format: str):
        self.expected_format = expected_format

    def __str__(self):
        return '\n'.join([
            f"Format error: end of file",
            f"\tExpected line with format '{self.expected_format}'",
            f"\tReceived END OF FILE"
        ])

class MismatchedRegexError(FormatError):
    def __init__(self, expected_format: str, actual_line: str, line_num: int):
        self.expected_format = expected_format
        self.actual_line = actual_line
        self.line_num = line_num

    def __str__(self):
        return '\n'.join([
            f"Format error: mismatched regex at line {self.line_num + 1}",
            f"\tExpected line with format '{self.expected_format}'",
            f"\tReceived '{self.actual_line}'"
        ])

class ConflictingValueError(FormatError):
    def __init__(self, k: str, v1: int, v2: int, line_num: int):
        self.k = k
        self.v1 = v1
        self.v2 = v2
        self.line_num = line_num

    def __str__(self):
        return '\n'.join([
            f"Format error: conflicting value assignment at line {self.line_num + 1}",
            f"\tKey '{self.k}' previously assigned with value '{self.v1}'",
            f"\tReceived conflicting assignment with value '{self.v2}'"
        ])

class ExtraDataError(FormatError):
    def __init__(self, num_extra_lines: int, line_num: int):
        self.num_extra_lines = num_extra_lines
        self.line_num = line_num

    def __str__(self):
        return '\n'.join([
            f"Format error: extra output at line {self.line_num + 1}",
            f"\tExpected END OF FILE",
            f"\tReceived {self.num_extra_lines} additional lines"
        ])

class A2_Data:
    def __init__(self):
        # Header values
        self.double_dict_len = 0
        self.triple_dict_len = 0
        self.all_tokens_len  = 0

        # Data
        self.double_dict    = dict()
        self.triple_dict    = dict()
        self.two_grams      = dict()
        self.sample_tokens  = set()
        self.dynamic_tokens = set()

    def check_header(self) -> List[str]:
        mismatches = []
        def check_mismatch(expected_len, actual_len, name) -> None:
            if expected_len != actual_len:
                mismatches.append(f"\tMismatch between declared {name} length ({expected_len}) and actual dictionary length ({actual_len})")

        check_mismatch(self.double_dict_len, len(self.double_dict), "double dictionary")
        check_mismatch(self.triple_dict_len, len(self.triple_dict), "triple dictionary")
        return mismatches

class A2_Parser:
    # Read header line
    def consume_header(data: A2_Data, strings: List[str], index: int) -> List[str]:
        HEADER_REGEX_STR = r'double dictionary list len (\d+), triple (\d+), all tokens (\d+)'
        HEADER_REGEX = re.compile(HEADER_REGEX_STR)

        # Check whether line exists
        if len(strings) == index:
            raise EndOfFileError(HEADER_REGEX_STR)

        # Match header
        header_line = HEADER_REGEX.fullmatch(strings[index])
        if header_line == None:
            raise MismatchedRegexError(HEADER_REGEX_STR, strings[index], index)

        data.double_dict_len = int(header_line.group(1))
        data.triple_dict_len = int(header_line.group(2))
        data.all_tokens_len  = int(header_line.group(3))

        return index + 1

    # Read dictionary section
    def consume_dict(output_dict, strings, index, header_str, entry_regex_str, array_element_regex_str):
        ENTRY_REGEX = re.compile(entry_regex_str)
        VALUE_REGEX = re.compile(array_element_regex_str)
        TERMINATOR  = r'---'

        # Check whether line exists
        if len(strings) == index:
            raise EndOfFileError(header_str)

        # Read dict header
        if strings[index] != header_str:
            raise MismatchedRegexError(header_str, strings[index], index)
        index += 1

        # Read dict data
        while index < len(strings) and strings[index] != TERMINATOR:
            entry = ENTRY_REGEX.fullmatch(strings[index])
            if entry == None:
                raise MismatchedRegexError(entry_regex_str, strings[index], index)

            # Assign count to each key
            count = int(entry.group(1))
            values = VALUE_REGEX.findall(entry.group(2))
            for value in values:
                while not isinstance(value, str):
                    value = value[0]

                if value in output_dict and output_dict[value] != count:
                    raise ConflictingValueError(value, count, output_dict[value], index)
                output_dict[value] = count

            index += 1

        # Check for terminator
        if len(strings) == index:
            raise MismatchedRegexError(terminator_str, strings[index], index)

        return index

    # Read double dictionary section
    def consume_double_dict(data, strings, index):
        TOKEN_REGEX = r'(\\.|[^\^"\\])+'
        DOUBLE_REGEX = r'"(' + TOKEN_REGEX + r'\^' + TOKEN_REGEX + r')"'
        index = A2_Parser.consume_dict(
            data.double_dict,
            strings,
            index,
            r'printing dict: double',
            r'(\d+): \[(' + DOUBLE_REGEX + r'(, ' + DOUBLE_REGEX + r')*)\]',
            DOUBLE_REGEX
        )

        return index + 1

    # Read triple dictionary section
    def consume_triple_dict(data, strings, index):
        TOKEN_REGEX = r'(\\.|[^\^"\\])+'
        TRIPLE_REGEX = r'"(' + TOKEN_REGEX + r'\^' + TOKEN_REGEX + r'\^' + TOKEN_REGEX + r')"'
        index = A2_Parser.consume_dict(
            data.triple_dict,
            strings,
            index,
            r'printing dict: triple',
            r'(\d+): \[(' + TRIPLE_REGEX + r'(, ' + TRIPLE_REGEX + r')*)\]',
            TRIPLE_REGEX
        )

        return index + 1

    # Read sample string tokens
    def consume_sample_tokens(data, strings, index):
        TOKEN_REGEX = r'(\\.|[^\^"\\])+'
        SAMPLE_TOKENS_REGEX_STR = r'\[("' + TOKEN_REGEX + r'"(, "' + TOKEN_REGEX + r'")*)?\]'
        SAMPLE_TOKENS_REGEX = re.compile(SAMPLE_TOKENS_REGEX_STR)
        VALUE_REGEX_STR = r'"(' + TOKEN_REGEX + r')"'
        VALUE_REGEX = re.compile(VALUE_REGEX_STR)

        # Check for empty line
        if len(strings) == index:
            raise EndOfFileError(SAMPLE_TOKENS_REGEX_STR)

        # Check for correct format
        sample_tokens = SAMPLE_TOKENS_REGEX.fullmatch(strings[index])
        if sample_tokens == None:
            raise MismatchedRegexError(SAMPLE_TOKENS_REGEX_STR, strings[index], index)

        # Read tokens
        values = VALUE_REGEX.findall(strings[index])
        for value in values:
            while not isinstance(value, str):
                value = value[0]

            data.sample_tokens.add(value)

        return index + 1

    # Read 2-grams
    def consume_two_grams(data, strings, index):
        TOKEN_REGEX = r'(\\.|[^\^"\\])+'
        DOUBLE_REGEX = r'(' + TOKEN_REGEX + r'\^' + TOKEN_REGEX + r')'
        TWO_GRAM_REGEX_STR = r'2-gram ' + DOUBLE_REGEX + r', count (\d+)'
        TWO_GRAM_REGEX = re.compile(TWO_GRAM_REGEX_STR)

        while index < len(strings):
            two_gram = TWO_GRAM_REGEX.fullmatch(strings[index])
            if two_gram == None: break

            two_gram_str = two_gram.group(1)
            count = int(two_gram.groups()[-1])

            if two_gram_str in data.two_grams and data.two_grams[two_gram_str] != count:
                raise ConflictingValueError(
                    two_gram_str,
                    data.two_grams[two_gram_str],
                    count,
                    index
                )
            data.two_grams[two_gram_str] = count
            index += 1

        return index

    # Read dynamic tokens
    def consume_dynamic_tokens(data, strings, index):
        TOKEN_REGEX = r'(\\.|[^\^"\\])+'
        DYNAMIC_TOKENS_REGEX_STR = r'dynamic tokens: \[("' + TOKEN_REGEX + r'"(, "' + TOKEN_REGEX + r'")*)?\]'
        DYNAMIC_TOKENS_REGEX = re.compile(DYNAMIC_TOKENS_REGEX_STR)
        VALUE_REGEX_STR = r'"(' + TOKEN_REGEX + r')"'
        VALUE_REGEX = re.compile(VALUE_REGEX_STR)

        # Check for empty line
        if len(strings) == index:
            raise EndOfFileError(DYNAMIC_TOKENS_REGEX_STR)

        # Check for correct format
        dynamic_tokens = DYNAMIC_TOKENS_REGEX.fullmatch(strings[index])
        if dynamic_tokens == None:
            raise MismatchedRegexError(DYNAMIC_TOKENS_REGEX_STR, strings[index], index)

        # Read values
        values = VALUE_REGEX.findall(''.join('' if s == None else s for s in dynamic_tokens.groups()))
        for value in values:
            while not isinstance(value, str):
                value = value[0]

            data.dynamic_tokens.add(value)

        return index + 1

    # Read whitespace
    def consume_whitespace(data, strings, index):
        WHITESPACE_REGEX = re.compile(r'\s*')
        while index < len(strings):
            whitespace = WHITESPACE_REGEX.fullmatch(strings[index])
            if whitespace == None: break
            index += 1
        return index

    # Parse entire output file
    def parse(file_name: str, strings: List[str]) -> A2_Data:
        data = A2_Data()
        index = 0

        try:
            # Parse data
            index = A2_Parser.consume_header        (data, strings, index)
            index = A2_Parser.consume_whitespace    (data, strings, index)
            index = A2_Parser.consume_double_dict   (data, strings, index)
            index = A2_Parser.consume_triple_dict   (data, strings, index)
            index = A2_Parser.consume_sample_tokens (data, strings, index)
            index = A2_Parser.consume_two_grams     (data, strings, index)
            index = A2_Parser.consume_dynamic_tokens(data, strings, index)
            index = A2_Parser.consume_whitespace    (data, strings, index)

            # Check for extra lines
            if len(strings) != index:
                raise ExtraDataError(len(strings) - index, index)
        except Exception as e:
            print(f"Encountered error while parsing file '{file_name}'")
            raise e

        return data

class A2_Comparator:
    def compare_headers(data1, data2) -> List[str]:
        mismatches = []
        def compare_header(name: str, v1: int, v2: int):
            if v1 != v2:
                mismatches.append(f"\tMismatch in declared {name}: {v1} vs. {v2}")

        compare_header("double_dict_len", data1.double_dict_len, data2.double_dict_len)
        compare_header("triple_dict_len", data1.triple_dict_len, data2.triple_dict_len)
        compare_header("all_tokens_len" , data1.all_tokens_len , data2.all_tokens_len)

        if len(mismatches) == 0:
            return []
        else:
            return ["Found mismatches in header:"] + mismatches + [""]

    def compare_dict(name, d1, d2, tolerance):
        mismatches = []
        keys = set(d1.keys()) | set(d2.keys())
        for key in keys:
            v1 = d1[key] if key in d1 else 0
            v2 = d2[key] if key in d2 else 0
            if abs(v1 - v2) > tolerance:
                mismatches.append(f"\t'{key}': {v1} vs. {v2}")

        if len(mismatches) == 0:
            return []
        else:
            return [f"Found mismatches exceeding tolerance for {name} with keys:"] + mismatches + [""]

    def compare_set(name, s1, s2):
        mismatches = []
        if s1 != s2:
            mismatches.append(f"{name} sets are not equal")
            diff1 = s1 - s2
            diff2 = s2 - s1

            if len(diff1) != 0: mismatches.append(f"\tSet 1 contains additional token(s) {diff1}")
            if len(diff2) != 0: mismatches.append(f"\tSet 2 contains additional token(s) {diff2}")
            mismatches += [""]
        return mismatches

    def compare(data1, data2, tolerance=0):
        mismatches = []
        mismatches += A2_Comparator.compare_headers(data1, data2)
        mismatches += A2_Comparator.compare_dict('double_dict'   , data1.double_dict   , data2.double_dict, tolerance)
        mismatches += A2_Comparator.compare_dict('triple_dict'   , data1.triple_dict   , data2.triple_dict, tolerance)
        mismatches += A2_Comparator.compare_dict('2-grams'       , data1.two_grams     , data2.two_grams  , tolerance)
        mismatches += A2_Comparator.compare_set ('sample token'  , data1.sample_tokens , data2.sample_tokens)
        mismatches += A2_Comparator.compare_set ('dynamic token' , data1.dynamic_tokens, data2.dynamic_tokens)
        return mismatches

if __name__ == '__main__':
    # Validate input
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(f"Usage: {sys.argv[0]} <FILE_1> <FILE_2> [TOLERANCE]")
        exit(1)

    # Read inputs
    filenames = [sys.argv[1], sys.argv[2]]
    try:
        tolerance = 0 if len(sys.argv) == 3 else int(sys.argv[3])
        if tolerance < 0: raise ValueError()
    except ValueError:
        print("TOLERANCE must be a non-negative integer value")
        exit(1)

    # Require input files to exist
    for filename in filenames:
        if not os.path.isfile(filename):
            print(f"Could not find file {filename}")
            exit(1)

    try:
        # Parse files
        data = [A2_Parser.parse(filename, open(filename).read().split('\n')) for filename in filenames]

        # Check for internal consistency
        mismatches = [datum.check_header() for datum in data]
        for i, filename in enumerate(filenames):
            mismatch = mismatches[i]
            if len(mismatch) != 0:
                print(f"Detected inconsistency in file '{filename}':")
                for s in mismatch:
                    print(f"{s}")
                print("")

        # Compare parsed data
        mismatches = A2_Comparator.compare(data[0], data[1], tolerance)
        if len(mismatches) == 0:
            print(f"Files match within specified tolerance")
        else:
            print(f"Detected mismatches while comparing files:")
            for mismatch in mismatches:
                print(f"\t{mismatch}")

    except FormatError as e:
        print(e)

    except KeyboardInterrupt:
        print("Aborted")