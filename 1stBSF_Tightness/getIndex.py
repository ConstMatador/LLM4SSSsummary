import struct


model_selected = "S2IPLLM"

def read_file(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            sequence = [struct.unpack('f', bytes.fromhex(line[i:i+8]))[0] for i in range(0, len(line.strip()), 8)]
            sequences.append(sequence)
    return sequences


def find_sequence_in_file(sequence, file2_sequences):
    for idx, seq in enumerate(file2_sequences):
        if seq == sequence:
            return idx
    raise ValueError(f"Sequence {sequence} not found.")


def main():
    file1_path = f"1stBSF_Tightness/{model_selected}/approSeries_1.txt"
    file2_path = f"1stBSF_Data/{model_selected}/reduce_data.bin"
    
    file1_sequences = read_file(file1_path)
    file2_sequences = read_file(file2_path)
    
    for idx, seq in enumerate(file1_sequences):
        try:
            position = find_sequence_in_file(seq, file2_sequences)
            print(f"Sequence {seq} from file1 found at position {position} in file2.")
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()
