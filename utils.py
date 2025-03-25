def print_wrapped(text, wrap_length=80):
    """
    Simple helper function that wraps and prints text to a certain 
    line length (useful for logging or debugging).
    """
    line = ""
    for word in text.split():
        if len(line) + len(word) + 1 > wrap_length:
            print(line)
            line = word
        else:
            if line:
                line += " "
            line += word
    if line:
        print(line)
