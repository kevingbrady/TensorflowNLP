import numpy as np
import pandas


def write_output(row, file, soft_matrix, catalog_length, sp_length, csf_length, index_list):

    matrix_row = np.array(soft_matrix[row.name])
    top_sp = matrix_row[catalog_length:(catalog_length + sp_length)].argsort()[-8:][::-1]
    top_csf = matrix_row[(catalog_length + sp_length):].argsort()[-5:][::-1]

    file.write(row['Sub-Element/Sub-Action'])
    file.write("\n"),
    file.write(str([index_list[catalog_length + ind] for ind in top_sp]))
    file.write('\n')
    file.write(str([matrix_row[catalog_length + ind] for ind in top_sp]))
    file.write('\n')
    file.write(str([index_list[catalog_length + sp_length + ind] for ind in top_csf]))
    file.write('\n')
    file.write(str([matrix_row[catalog_length + sp_length + ind] for ind in top_csf]))
    file.write("\n\n\n")


def write_catalog_row(row, file):
    file.write('-----------------------------------------------------------------\n')
    file.write("[NON-TECHNICAL] ")
    file.write("NaN" if pandas.isna(row['NON-TECHNICAL']) else row['NON-TECHNICAL'])
    file.write("\n")
    file.write("[Sub-Capability] ")
    file.write("NaN" if pandas.isna(row['Sub-Capability']) else row['Sub-Capability'])
    file.write("\n")
    file.write("[Element/Action] ")
    file.write("NaN" if pandas.isna(row['Element/Action']) else row['Element/Action'])
    file.write("\n")
    file.write("[Sub-Element/Sub-Action] ")
    file.write("NaN" if pandas.isna(row['Sub-Element/Sub-Action']) else row['Sub-Element/Sub-Action'])
    file.write("\n")
    file.write('-----------------------------------------------------------------\n\n')
    

def print_run_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2} hrs {:0>2} min {:05.2f} s".format(int(hours), int(minutes), int(seconds))


