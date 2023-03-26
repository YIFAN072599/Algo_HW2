import shutil
import os

WORK_DIR = os.path.dirname(__file__)
tickers = ['MS', 'AAPL', 'MSFT', 'AMZN', 'JPM']
date = ['20070709', '20070710', '20070711', '20070712', '20070713', '20070716']
src_dir = '/Users/chenzhao/Data/taq data/trades'


def copy_file_to_folder(src_file_path, dest_folder_path):
    """
    Copy a specific file to a new folder.

    Parameters:
    - src_file_path (str): The path of the file to be copied.
    - dest_folder_path (str): The path of the folder to copy the file to.

    Returns:
    - None
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder_path):
        os.makedirs(dest_folder_path)

    # Use shutil to copy the file to the new folder
    shutil.copy(src_file_path, dest_folder_path)

    print(f"File {src_file_path} has been copied to {dest_folder_path}")


if __name__ == '__main__':
    for d in date:
        dest_folder_path = os.path.join(WORK_DIR, 'data', 'trades', d)
        for t in tickers:
            src_file_path = os.path.join(src_dir, d, t+'_trades.binRT')
            copy_file_to_folder(src_file_path, dest_folder_path)
