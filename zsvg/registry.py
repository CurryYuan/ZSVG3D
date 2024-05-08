# This module will contain the registry and the decorator

# Define a registry
interpreter_registry = {}

# Define a decorator for registering classes
def register_interpreter(interpreter):
    step_name = interpreter.step_name
    if isinstance(step_name, list):
        for name in step_name:
            interpreter_registry[name] = interpreter()
    else:
        interpreter_registry[step_name] = interpreter()
    return interpreter
