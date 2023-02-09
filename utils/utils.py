import os


"""
Function for create a directory if it does not exist
"""
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Directory created!")
    else:
        print("Directory already created")
