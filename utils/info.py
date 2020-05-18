import pandas as pd

def get_seiz_types():
    seiz_types_path = '/media/przemek/08041BCD041BBC9E/tuh_eeg_seizure/v1.5.1/_DOCS/seizures_types_v02.xlsx'
    seiz_types = pd.read_excel(seiz_types_path)
    seiz_types = seiz_types.set_index('Class Code')
    return seiz_types


def print_seiz_types():
    seiz_types = get_seiz_types()
    with pd.option_context('display.max_rows', 30, 'display.max_columns', 5):
        print(seiz_types)

def print_train_seiz_types_summaries():
    seiz_types = get_seiz_types()
    train_info = load_seiz_info_file()
    train_seiz_type = train_info.iloc[1:12, 26:30]
    train_seiz_type.columns = ['Class Code', 'Events', 'Freq.', 'Cum.']
    train_seiz_type = train_seiz_type.set_index('Class Code')
    train_seiz_type.join(seiz_types)
    print(train_seiz_type)


def print_train_class_summaries():
    train_info = load_seiz_info_file()
    train_class_summary = train_info.iloc[9:12, 16:20]
    train_class_summary.columns = ['Normal Classification', 'Sessions', 'Freq.', 'Cum.']
    train_class_summary = train_class_summary.set_index('Normal Classification')
    print(train_class_summary)


def get_data_info(data_type='train'):
    # seiz_info_path = '/media/przemek/Shared Files/tuh_eeg_seizure/v1.5.1/_DOCS/seizures_v34r.xlsx'
    info = load_seiz_info_file(sheet_name=data_type)
    # just want the utils per file here
    file_info = info.iloc[1:6101, 1:15]
    # cleans some of the names
    file_info_cols = ['File No.', 'Patient', 'Session', 'File',
                      'EEG Type', 'EEG SubType', 'LTM or Routine',
                      'Normal/Abnormal', 'No. Seizures File',
                      'No. Seizures/Session', 'Filename', 'Seizure Start',
                      'Seizure Stop', 'Seizure Type']
    file_info.columns = file_info_cols

    # we forward fill as there are gaps in the excel file to represent the utils
    # is the same as above (apart from in the filename, seizure start, seizure stop
    # and seizure type columns)
    for col_name in file_info.columns[:-4]:
        file_info[col_name] = file_info[col_name].ffill()

    # patient ID is an integer rather than float
    file_info['Patient'] = file_info['Patient'].astype(int)
    return file_info


def print_patients_info_head():
    file_info = get_data_info()
    with pd.option_context('display.max_rows', 5, 'display.max_columns', 10):
        print(file_info.head())


def load_seiz_info_file(sheet_name='train'):
    seiz_info_path = '/media/przemek/Shared Files1/tuh_eeg_seizure/v1.5.1/_DOCS/seizures_v34r.xlsx'
    seiz_info = pd.read_excel(seiz_info_path, sheet_name)
    return seiz_info



