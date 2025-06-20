import pandas as pd
import os
import time
from datetime import datetime

from pandas import pivot

from _dataTreatment.models.curingas import list_folders_to_menu, sums_calculate, get_split_name, list_files_2_menu

ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)

"""
    Tem algum bug neste código. Eu preciso rodar ele 2 vezes pra coletar todos os valores dos dataframes.
    Procurar e ajustar.
"""


def format_datetime():
    """Returns the current datetime in a format suitable for filenames."""
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def get_media(path, name, param):

    # print(param)

    llm = param['llm']
    version = param['version']
    refactored_project = param['project']

    split_pop = path.split('/')
    split_pop.pop()
    out_file = '/'.join(split_pop)
    df = pd.DataFrame()
    csv_file_path = pd.read_csv(path)
    df = pd.concat([df, csv_file_path])
    df = df.drop(df.columns[0], axis=1)
    means = df.mean()

    df2 = pd.DataFrame(means)
    dft = df2.T

    dft.to_csv(f'{out_file}/means-{refactored_project}.{llm}.{version}-{name}.csv', index=False)
    # dft.to_csv(f'{out_file}/means-{format_datetime()}-{name}.csv', index=False)


def separe_dataframe(df, path, param):
    df.set_index('Unnamed: 0', inplace=True)
    grouped = df.groupby(level=0)
    dataframes = {name: group.reset_index() for name, group in grouped}

    for name, df_group in dataframes.items():
        df_group.to_csv(f"{path}-{name}.csv", index=False)
        get_media(f'{path}-{name}.csv', name, param)


def aggregate(path):
    df = pd.DataFrame()
    for file in os.listdir(path):
        csv_file_path = pd.read_csv(os.path.join(path, file))
        df = pd.concat([df, csv_file_path])
    df.to_csv(f'{file}', index=False)


def generate_aggregate_csv(path, param):
    llm = param['llm']
    version = param['version']
    project = param['project']

    output_aggregated = f'{path}/aggregated'
    if not os.path.exists(output_aggregated):
        os.makedirs(output_aggregated)

    output_aggregated_file_path = f'{output_aggregated}/aggregated-{project}.{llm}.{version}'

    df = pd.DataFrame(columns=['Unnamed: 0'])
    # df = pd.DataFrame(columns=['Metric'])
    for file in os.listdir(path):
        if file == 'aggregated':
            continue
        else:
            csv_file_path = pd.read_csv(os.path.join(path, file))
            df = pd.concat([df, csv_file_path])
            csv_file_path.drop(columns=df.columns[0], axis=1, inplace=True)
            df.to_csv(f'{output_aggregated_file_path}.csv', index=False)
    separe_dataframe(df, output_aggregated_file_path, param)


def contains_test(cell_value):
    if isinstance(cell_value, str):
        substrings = ['test', '-test', '_test', '_tests', '-tests']
        if any(substring in cell_value for substring in substrings) or cell_value.startswith(('test/', 'test',
                                                                                              '-test', '_test',
                                                                                              '_tests', '-tests')):
            return True
    return False


def get_sums_columns(dataframe, output_file_path, param, flag):
    if flag:
        print('[ATENÇÃO] - Dataframe original...')
        dataframe = pd.read_csv(dataframe)
        dataframe = dataframe[~dataframe.iloc[:, 0].apply(contains_test)]
        resulted = {}

        resulted_missing = {}
        for column in dataframe.select_dtypes(include='number'):
            if column == 'statements_diff':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column][:-1])
                resulted[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file = f'{output_file_path}/{param}_statements.csv'
                sums_results_df = pd.DataFrame.from_dict(resulted, orient='index')
                sums_results_df.to_csv(output_file)

            elif column == 'missing_diff':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column][:-1])
                resulted_missing[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file_missing = f'{output_file_path}/{param}_missing.csv'
                sums_results_df = pd.DataFrame.from_dict(resulted_missing, orient='index')
                sums_results_df.to_csv(output_file_missing)

            elif column == 'LINE_COVERED':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column][:-1])
                resulted[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file = f'{output_file_path}/{param}_LINE_COVERED.csv'
                sums_results_df = pd.DataFrame.from_dict(resulted, orient='index')
                sums_results_df.to_csv(output_file)

            elif column == 'LINE_MISSED':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column][:-1])
                resulted_missing[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file_missing = f'{output_file_path}/{param}_LINE_MISSED.csv'
                sums_results_df = pd.DataFrame.from_dict(resulted_missing, orient='index')
                sums_results_df.to_csv(output_file_missing)

    else:
        dataframe = dataframe[:-1]
        dataframe = dataframe[~dataframe.iloc[:, 0].apply(contains_test)]

        smell_name      = get_split_name(output_file_path, '.', -2)
        llm_model       = param['llm']
        version_number  = param['version']
        project_name    = param['project']
        language        = param['language']

        output_file_path = (
            f'{parent_dir}/{language}/coverage/data_analysis/_total/refactored/optimization/'
            f'{llm_model}/{version_number}/{project_name}')
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

        resulted = {}
        resulted_missing = {}
        for column in dataframe.select_dtypes(include='number'):
            if column == 'statements_diff':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column])
                resulted[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': negative_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': positive_sum
                }
                output_file = (f'{output_file_path}/{project_name}.{llm_model}.{version_number}.{smell_name}'
                               f'_statements_results.csv')
                generate_aggregate_csv(output_file_path, param)
                sums_results_df = pd.DataFrame.from_dict(resulted, orient='index')
                sums_results_df.to_csv(output_file)

            if column == 'missing_diff':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column])
                resulted_missing[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': negative_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': positive_sum
                }
                output_file_missing = (f'{output_file_path}/{project_name}.{llm_model}.{version_number}.{smell_name}'
                                       f'_missing_results.csv')
                generate_aggregate_csv(output_file_path, param)
                sums_results_df = pd.DataFrame.from_dict(resulted_missing, orient='index')
                sums_results_df.to_csv(output_file_missing)


            if column == 'LINE_COVERED':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column])
                resulted[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file = (f'{output_file_path}/{project_name}.{llm_model}.{version_number}.{smell_name}'
                               f'_LINE_COVERED_results.csv')
                generate_aggregate_csv(output_file_path, param)
                sums_results_df = pd.DataFrame.from_dict(resulted, orient='index')
                sums_results_df.to_csv(output_file)

            if column == 'LINE_MISSED':
                total, positive_sum, negative_sum = sums_calculate(dataframe[column])
                resulted_missing[column] = {
                    'total': total,
                    # Total sum of positive values
                    'lost_coverage_lines': positive_sum,
                    # Total sum of negative values
                    'improve_coverage_lines': negative_sum
                }
                output_file_missing = (f'{output_file_path}/{project_name}.{llm_model}.{version_number}.{smell_name}'
                                       f'_LINE_MISSED_results.csv')
                generate_aggregate_csv(output_file_path, param)
                sums_results_df = pd.DataFrame.from_dict(resulted_missing, orient='index')
                sums_results_df.to_csv(output_file_missing)


def get_original_project_sum():
    original_file_sum = input("Get original project sum? (y/n)\n")
    return original_file_sum.lower() == 'y'


def process_original_project_sum(language, dataset_abs_path, values_test):
    global parent_dir

    original_project_file_path = list_files_2_menu(
        os.path.join(language, dataset_abs_path), word='Project'
    )
    original_project_name = original_project_file_path.split('/')[-1].replace("(Original_File)-", "").replace(".csv", "")

    values_test['project'] = original_project_name

    output_file_path = os.path.join(
        parent_dir,
        language.split('/')[-1],
        'coverage/data_analysis/_total/originals/'
    )

    os.makedirs(output_file_path, exist_ok=True)

    get_sums_columns(original_project_file_path, output_file_path, values_test['project'], flag=True)


def compare_and_calculate_difference(original, refactored, output_file, params, language):

    if language.split('/')[-1] == 'python':
        original_df = pd.read_csv(original)
        original_df = original_df[:-1]
        original_df = original_df[~original_df.iloc[:, 0].apply(contains_test)]

        refactored_df = pd.read_csv(refactored)
        refactored_df = refactored_df[:-1]
        refactored_df = refactored_df[~refactored_df.iloc[:, 0].apply(contains_test)]

        key_column = original_df.columns[0]
        merged_df = pd.merge(original_df, refactored_df, how='inner', on=key_column)

        # Initialize a DataFrame to store the differences
        diff_df = merged_df[[key_column]].copy()
        df_diff = diff_df.copy()

        print(f'Obtain Python difference...')

        df_diff['missing_orig']         = merged_df['missing_x']
        df_diff['missing_refac']        = merged_df['missing_y']
        df_diff['statements_orig']      = merged_df['statements_x']
        df_diff['statements_refac']     = merged_df['statements_y']
        df_diff['statements_diff']      = merged_df['statements_y']     - merged_df['statements_x']
        df_diff['missing_diff']         = merged_df['missing_y']        - merged_df['missing_x']
        df_diff['coverage_original']    = merged_df['statements_x']     - merged_df['missing_x']
        df_diff['coverage_refactor']    = merged_df['statements_y']     - merged_df['missing_y']
        df_diff['covered_lines']        = df_diff['coverage_refactor']  - df_diff['coverage_original']

        # Adiciona a linha de total ao DataFrame
        total_row = {
            'missing_orig':         df_diff['missing_orig'].sum(),
            'missing_refac':        df_diff['missing_refac'].sum(),
            'statements_orig':      df_diff['statements_orig'].sum(),
            'statements_refac':     df_diff['statements_refac'].sum(),
            'statements_diff':      df_diff['statements_diff'].sum(),
            'missing_diff':         df_diff['missing_diff'].sum(),
            'coverage_original':    df_diff['coverage_original'].sum(),
            'coverage_refactor':    df_diff['coverage_refactor'].sum(),
            'covered_lines':        df_diff['covered_lines'].sum(),
            'Module': 'Total'
        }

        df_diff = pd.concat([df_diff, pd.DataFrame([total_row])], ignore_index=True)
        get_sums_columns(df_diff, output_file, params, flag=False)
        df_diff.to_csv(output_file, index=False)

    elif language.split('/')[-1] == 'java':
        original_df = pd.read_csv(original)
        original_df.drop(columns=["GROUP", "PACKAGE"], inplace=True)
        refactored_df = pd.read_csv(refactored)
        refactored_df.drop(columns=["GROUP", "PACKAGE"], inplace=True)

        result_rows = []
        for i, row in original_df.iterrows():
            ref_row = refactored_df[refactored_df['CLASS'] == row['CLASS']]
            if not ref_row.empty:
                statements_diff = ref_row['LINE_COVERED'].values[0] - row['LINE_COVERED']
                missing_diff    = ref_row['LINE_MISSED'].values[0]  - row['LINE_MISSED']
                result_rows.append(
                    {'CLASS': row['CLASS'], 'LINE_COVERED': statements_diff, 'LINE_MISSED': missing_diff})
            else:
                missing_sum = row['LINE_COVERED'] + row['LINE_MISSED']
                result_rows.append({'CLASS': row['CLASS'], 'LINE_COVERED': 0, 'LINE_MISSED': missing_sum})


        df_diff = pd.DataFrame(result_rows)
        get_sums_columns(df_diff, output_file, params, flag=False)
        df_diff.to_csv(output_file, index=False)


def normalize_key_column(dataframe) -> pd.DataFrame:

    column_names = ['File', 'Module']
    df = pd.read_csv(dataframe)

    """
    Normaliza a primeira coluna de um DataFrame para um nome padrão.

    Args:
        df (pd.DataFrame): O DataFrame a ser processado.
        column_names (list): Lista de nomes possíveis para a primeira coluna.

    Returns:
        pd.DataFrame: DataFrame com a coluna chave normalizada.
    """

    for col in column_names:
        if col in df.columns:
            df = df.rename(columns={col: "File"}, inplace=True)
            break

    return df


def main():
    values_test = {}
    # flag = False
    dataset_abs_path = 'coverage/extraction_results/_dataset_original'

    # language = list_folders_to_menu(parent_dir, word='Language')
    # extraction_results_path = os.path.join(parent_dir, language, 'coverage/extraction_results')
    # output_folder = os.path.join(parent_dir, language, 'coverage/data_analysis/_difference')

    # Gerar somas dos arquivos originais
    if get_original_project_sum():
        process_original_project_sum(language, dataset_abs_path, values_test)


    else:
        approaches = list_folders_to_menu(extraction_results_path, word='Approach')
        llms = list_folders_to_menu(approaches, word='LLM')
        versions = list_folders_to_menu(llms, word='Version')

        approach = get_split_name(approaches, '/', -1)
        llm = get_split_name(llms, '/', -1)
        version = get_split_name(versions, '/', -1)

        refactored_projects = os.listdir(versions)
        original_files = os.listdir(os.path.join(language, dataset_abs_path))

        for original_project in original_files:
            original_file_name = (original_project.split('/')[-1].replace("(Original_File)-", "").
                                  replace(".csv", ""))
            for refactored_project in refactored_projects:
                if refactored_project == original_file_name:
                    values_test = {
                        'llm': llm,
                        'version': version,
                        'project': refactored_project,
                        'language': language.split('/')[-1],
                    }

                    """
                        function to control whether the original file name is identical to the refactored project
                        name project: receives a list of all projects that contain release folders
                    """
                    csv_projects_list = [folder for folder in os.listdir(os.path.join(versions, refactored_project))]

                    output_path = f'{output_folder}/{approach}/{llm}/{version}/{refactored_project}'
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    for file in csv_projects_list:
                        output_file = f'{output_path}/difference_{file}'
                        refactored_project_file_path = os.path.join(versions, refactored_project, file)
                        original_project_file_path = os.path.join(language, dataset_abs_path, original_project)
                        compare_and_calculate_difference(original_project_file_path,
                                                         refactored_project_file_path,
                                                         output_file,
                                                         values_test,
                                                         language)

                if refactored_project == 'bySmell':
                    folders_by_smell = os.listdir(os.path.join(versions, refactored_project))
                    for smell in folders_by_smell:
                        smell_dir = os.path.join(versions, refactored_project, smell)
                        for file in os.listdir(smell_dir):
                            project_name_by_smell = file.split('.')[0]
                            if project_name_by_smell == original_file_name:
                                refactored_project_file_path = os.path.join(smell_dir, file)
                                values_test = {
                                    'llm': llm,
                                    'version': version,
                                    'project': refactored_project,
                                    'language': language.split('/')[-1],
                                }

                                """
                                    function to control whether the original file name is identical to the refactored project
                                    name project: receives a list of all projects that contain release folders
                                """

                                output_path = f'{output_folder}/{approach}/{llm}/{version}/{refactored_project}/{smell}'
                                if not os.path.exists(output_path):
                                    os.makedirs(output_path)

                                output_file = f'{output_path}/difference_{file}'
                                original_project_file_path = os.path.join(language, dataset_abs_path, original_project)
                                # print(original_project_file_path)

                                compare_and_calculate_difference(original_project_file_path,
                                                                 refactored_project_file_path,
                                                                 output_file,
                                                                 values_test,
                                                                 language)
                            # print(project_name_by_smell, original_file_name)

    # else:
    #     process_refactored_projects(language, dataset_abs_path, extraction_results_path, output_folder, values_test)

if __name__ == "__main__":
    language = list_folders_to_menu(parent_dir, word='Language')
    extraction_results_path = os.path.join(parent_dir, language, 'coverage/extraction_results')
    output_folder = os.path.join(parent_dir, language, 'coverage/data_analysis/_difference')
    main()

    print('Done!!!')

