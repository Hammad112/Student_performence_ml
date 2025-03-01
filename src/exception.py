import sys
from src.logger import logging

## Function to create custom exception
def error_message_detail(error, error_detail:sys):
    ## error_detail.exc_info() will give threee important values
    ## first one is error, second one is object and third one is traceback
    ## We are interested in third one
    ## which line of code is causing the error, and error message

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
