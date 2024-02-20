def is_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            raise ImportError("Not in IPython")
        return True
    except Exception:
        return False

def dynamic_print(string_to_print):
    """
    Dynamically update terminal printout.

    Parameters
    ----------
    *args : str
        Strings to print

    """
    formatted_string = f"\r{string_to_print} " # small space after to make it look better in terminal
    print(formatted_string, end='', flush=True)
