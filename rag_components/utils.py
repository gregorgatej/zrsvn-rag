import textwrap

def print_wrapped(
    text, 
    wrap_length=80
):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)