import os

def extract_elements_and_save_new_txt(directory, out_directory):
    os.makedirs(out_directory, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            new_lines = []
            with open(os.path.join(directory, filename), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.strip().startswith(('0', '1')):
                        # 将原行进行分割
                        elements = line.split()
                        # 对除了第一个元素外的每个元素，将其转化为浮点数并保留6位小数，之后再转回字符串
                        elements[1:] = [format(float(e), '.6f') for e in elements[1:]]
                        # 将处理后的元素合并回一行
                        new_line = ' '.join(elements) + '\n'
                        new_lines.append(new_line)

            with open(os.path.join(out_directory, filename), 'w') as out_file:
                out_file.writelines(new_lines)

input_label = 'F:\doctor\distill_fundation_model\output_1_1'
output_label = 'F:\doctor\distill_fundation_model\data\labels_1_1'
files = os.listdir(input_label)
for file in files:
    label_path = os.path.join(input_label, file)
    output_path = os.path.join(output_label, file)
    extract_elements_and_save_new_txt(label_path, output_path)
