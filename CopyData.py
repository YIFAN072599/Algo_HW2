import os
import shutil


from CollectTicker import collect_ticker, collect_date

WORK_DIR = os.path.dirname(__file__)
tickers = collect_ticker()
date = collect_date()
src_dir = "C:\\Users\\Admin\\Downloads\\Algo HW2\\Algo HW2\\trades\\trades"


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
            src_file_path = os.path.join(src_dir, d, str(t) + '_trades.binRT')
            try:
                copy_file_to_folder(src_file_path, dest_folder_path)
            except FileNotFoundError as e:
                print(e)
