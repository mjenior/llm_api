import os
import re
import sys

from promptpal.lib import text_library

patternDict = text_library["patterns"]


def setup_logging(prefix):
    """
    Prepare logging setup.
    """
    log_file = utils.check_unique_filename(f"logs/{prefix}.transcript.log")
    os.makedirs("logs", exist_ok=True)
    with open(log_file, "w") as f:
        f.write("New session initiated.\n")

    return log_file


def string_to_binary(input_string):
    """Create a binary-like variable from a string for use a random seed"""
    # Convert all characters in a str to ASCII values and then to 8-bit binary
    binary = "".join([format(ord(char), "08b") for char in input_string])

    return int(binary[0 : len(str(sys.maxsize))])  # Constrain length


def _is_code_file(file_path):
    """Check if a file has a code extension."""
    return os.path.splitext(file_path)[1].lower() in set(extDict.values())


def check_unique_filename(filename):
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    counter = 1
    # Check if the file exists, and modify the name if it does
    while os.path.exists(filename):
        filename = f"{name}_{counter}{ext}"
        counter += 1

    return filename


def extract_object_names(code, language):
    """
    Extract defined object names (functions, classes, and variables) from a code snippet.
    """
    # Get language-specific patterns
    patterns = patternDict.get(language, {})

    # Extract object names using the language-specific patterns
    classes = patterns.get("class", re.compile(r"")).findall(code)
    functions = patterns.get("function", re.compile(r"")).findall(code)
    variables = patterns.get("variable", re.compile(r"")).findall(code)

    # Select objects to return based on hierarachy
    if len(classes) > 0:
        return classes
    elif len(functions) > 0:
        return functions
    else:
        return variables


def find_max_lines(code, object_names):
    """
    Count the number of lines of code for each object in the code snippet.

    Args:
        code (str): The code snippet to analyze.
        object_names (list): A list of object names to count lines for.

    Returns:
        str: Name of object with the largest line count.
    """
    rm_names = ["main", "functions", "classes", "variables"]
    line_counts = {name: 0 for name in object_names if name not in rm_names}
    line_counts["code"] = 1
    current_object = None

    for line in code.split("\n"):
        # Check if the line defines a new object
        for name in object_names:
            if re.match(rf"\s*(def|class)\s+{name}\s*[\(:]", line):
                current_object = name
                break

        # Count lines for the current object
        if current_object and line.strip() and current_object not in rm_names:
            line_counts[current_object] += 1

    return max(line_counts, key=line_counts.get)


def validate_probability_params(temp, topp):
    """Ensure temperature and top_p are valid"""
    temp = 0.7 if not (0.0 <= temp <= 2.0) else temp
    topp = 1.0 if not (0.0 <= topp <= 2.0) or temp != 0.7 else topp
    return temp, topp


def read_file_contents(filename):
    """Reads the contents of a given file."""
    with open(filename, "r", encoding="utf-8") as f:
        return f"# File: {filename}\n{f.read()}"


def scan_directory(path="code"):
    """Recursively scan a directory and return the content of all code files."""
    codebase = ""
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if _is_code_file(file_path):
                codebase += f"File: {file_path}\n"
                codebase += read_file_contents(file_path) + "\n\n"

    return codebase


def find_existing_paths(prompt):
    """Scan the input string for existing paths and return them in separate lists."""
    # Regular expression to match potential file paths
    path_pattern = re.compile(r'([a-zA-Z]:\\[^:<>"|?\n]*|/[^:<>"|?\n]*)')

    # Find all matches in the input string
    matches = path_pattern.findall(prompt)

    # Separate files and directories
    existing_paths = []
    for match in matches:
        if os.path.isdir(match):
            existing_paths.append(match)

    return existing_paths


def find_existing_files(prompt):
    """Filter filenames by checking if they exist in the current directory or system's PATH"""
    existing_files = [
        x for x in prompt.split() if os.path.isfile(x.rstrip(string.punctuation))
    ]
    return existing_files


def extract_code_snippets(message):
    """Extract code snippets from a large body of text using triple backticks as delimiters."""
    # Regular expression to match code blocks enclosed in triple backticks, including the language tag
    code_snippets = defaultdict(str)
    code_pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
    snippets = code_pattern.findall(message)
    for lang, code in snippets:
        code_snippets[lang] += code.strip()

    return code_snippets
