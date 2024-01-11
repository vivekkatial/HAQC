"""
Utility functions for running experiments

Author: Vivek Katial
"""

import argparse
import contextlib
import tempfile
import shutil
import re
import boto3
from botocore.exceptions import NoCredentialsError


def check_boto3_credentials():
    try:
        # Create a boto3 client to check for credentials
        client = boto3.client('s3')
        # Try to list buckets as a test command, it requires valid credentials
        client.list_buckets()
        return "AWS credentials are found and valid."
    except NoCredentialsError:
        return "No AWS credentials found."
    except Exception as e:
        return f"An error occurred: {e}"


def to_snake_case(string):
    string = string.replace("3", "three")
    string = string.replace("4", "four")
    string = (
        re.sub(r'(?<=[a-z])(?=[A-Z])|[^a-zA-Z]', ' ', string).strip().replace(' ', '_')
    )
    return ''.join(string.lower())


def str2bool(v):
    """Function to convert argument into ArgParse to be boolean
    :param v: Input from user
    :type v: str
    :returns: True or False of boolean type
    :rtype: {bool}
    :raises: argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


@contextlib.contextmanager
def make_temp_directory():
    """Make temp directory"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def clean_parameters_for_logging(algo_result, **kwargs):
    """Clean Algorithm result parameter from QAOA to log on MLFlow"""
    params = algo_result.optimal_parameters
    # Convert from ParameterVectorElement to simple string
    cleaned_params = {k.name: v for k, v in params.items()}
    # Replace greek with alpha-numeric
    cleaned_params = {k.replace('β', 'beta'): v for k, v in cleaned_params.items()}
    cleaned_params = {k.replace('γ', 'gamma'): v for k, v in cleaned_params.items()}
    cleaned_params = {k.replace('[', '_'): v for k, v in cleaned_params.items()}
    cleaned_params = {k.replace(']', ''): v for k, v in cleaned_params.items()}

    kwarg_str = ''
    for key, value in kwargs.items():
        kwarg_str += f'{key}_{str(value)}_'

    cleaned_params = {f"{kwarg_str}_{k}": v for k, v in cleaned_params.items()}
    return cleaned_params
