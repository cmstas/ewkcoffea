import csv
        
def save_array_to_csv(arr, file_name):
    import csv
    """
    Save a 2D array to a CSV file.
    Assumes arr[0] is the header row, arr[1:] are data rows.
    """
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row)