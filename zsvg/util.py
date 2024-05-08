import logging
import os
import io, tokenize
import numpy as np
from icecream import ic


def parse_step(step_str, partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(output_var=output_var, step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = tokens[4:-3]
    args = dict()
    i = 0
    while i < len(arg_tokens):
        key = arg_tokens[i].string
        if arg_tokens[i + 1].string == '=':
            value_token = arg_tokens[i + 2]
            if value_token.string == '[':     # Handle list arguments
                list_values = []
                i += 3     # Move to the next token after the '=' and '['
                while arg_tokens[i].string != ']':
                    if arg_tokens[i].string != ',':
                        list_values.append(arg_tokens[i].string)
                    i += 1     # Move to the next token within the list
                value = list_values
                i += 1     # Move past the closing ']'
            else:
                value = value_token.string
                i += 2     # Move to the next key-value pair
        args[key] = value
        i += 1     # Move to the next key in the key-value pair

        # Skip the comma token outside of lists
        if i < len(arg_tokens) and arg_tokens[i].string == ',':
            i += 1

    parsed_result['args'] = args
    return parsed_result


def create_logger(exp_name):
    # Create a custom logger
    logger = logging.getLogger('my_logger')

    # Set the log level of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create handlers for writing to a file and logging to console
    file_handler = logging.FileHandler(f'logs/{exp_name}.log', mode='w')
    console_handler = logging.StreamHandler()

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths
    Returns:
        iou
    """

    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0 * intersection / union