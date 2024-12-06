# Function to read .dat file and parse data
def read_dat_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split() # reading the 5th column
            data.append((line.strip(), float(columns[4]))) # writing the line + 5th column
    return data
# function to compare the data
def compare_and_write(file1, file2, output_file, threshold=0.1):
    data1 = read_dat_file(file1)
    data2 = read_dat_file(file2)

    with open(output_file, 'w') as out_file:
        for i, (_, value1) in enumerate(data1): #reaing 5th column as value 1
            line2, value2 = data2[i] # reading line + column 
            if abs(value1 - value2) > threshold: # write if >= needed
                out_file.write(f"{line2}\n")

    print(f"written {output_file} with difference greater than {threshold}")


# File paths
file1 = "data1.dat" # change input file 1 name
file2 = "data2.dat" # change input file 2 name
output_file = "data3.dat" # change output file name

compare_and_write(file1, file2, output_file)
