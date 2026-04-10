import pandas as pd
import matplotlib.pyplot as plt


s1=pd.read_csv('s1.txt')
s2=pd.read_csv('s2.txt')
s3=pd.read_csv('s3.txt')

new_s1 = s1.dropna()
print(new_s1.to_string())












class coord:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"X={self.x}, Y={self.y})"


# --- MADDE 1: Parser Function ---
def parse_tsp_file(filename):
    data = []
    try:
        with open(filename, 'r') as file: #read the file
            lines = file.readlines() # read the all of lines
            for line in lines:
                parts = line.strip().split() # split the element 1. count 2.x coord 3. y coord
                if len(parts) >= 2:
                    new_coord = coord(parts[0], parts[1]) #parts[0] parts[1] parts[2] = count x coord y coord
                    data.append(new_coord) # added to cities list to new city
        print(f"SUCCESSFUL: Loaded {len(data)} cities from file {filename}.")
        return data
    except Exception as e: #something wrong print error
        print(f"ERROR: {e}")
        return []




if __name__ == "__main__":
    files_to_test = ["s1.txt"]
    for file_name in files_to_test:
        coord2 = parse_tsp_file(file_name)
        c1 = coord2[0]
        c2 = coord2[1]
        print(f"coord x {c1}, y {c1}")











































