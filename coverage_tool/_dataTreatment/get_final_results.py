import pandas as pd
from _dataTreatment.models.curingas import list_folders_to_menu, get_split_name
import os

ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)

results_path = 'coverage/data_analysis/_total'


def add_mean_row(input_csv):
    df = pd.read_csv(input_csv)

    means = df.mean(numeric_only=True)
    sums = df.sum(numeric_only=True)

    means['project_name'] = 'média'
    sums['project_name'] = 'sum'

    # Adicionar a linha de médias ao DataFrame
    df_means = pd.DataFrame(means).transpose()
    df_sums = pd.DataFrame(sums).transpose()
    df = pd.concat([df, df_means, df_sums], ignore_index=True)
    df.to_csv(input_csv, index=False)


if __name__ == "__main__":

    language = list_folders_to_menu(parent_dir, word='Language')
    data_base = list_folders_to_menu(os.path.join(language, results_path), word='DataBase')
    approach = list_folders_to_menu(data_base, word='Approach')
    llm_name = list_folders_to_menu(approach, word='LLM')
    version_number = list_folders_to_menu(llm_name, word='Version')

    llm = get_split_name(llm_name, '/', -1)
    version = get_split_name(version_number, '/', -1)
    language = get_split_name(language, '/', -1)

    combined_missing = []
    combined_statements = []
    combined_line_missed = []
    combined_line_covered = []

    output_dir = f'{parent_dir}/{language}/coverage/results/{llm}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(version_number)
    for file in files:
        if file != 'bySmell':
            df = pd.DataFrame(columns=['Projetc'])
            aggregated = os.listdir(os.path.join(version_number, file))
            for data in aggregated:

                if os.path.isdir(os.path.join(version_number, file, data)):
                    for csv in os.listdir(os.path.join(version_number, file, data)):
                        # print(csv)
                        if csv.startswith('means') and csv.endswith('missing_diff.csv'):
                            # print(csv)
                            df = pd.read_csv(os.path.join(version_number, file, data, csv))
                            df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
                            combined_missing.append(df)  # adiciona os novos items ao Final da lista
                            combined_df = pd.concat(combined_missing, ignore_index=True)  # concatena a lista
                            combined_df.to_csv(f'{output_dir}/{version}-{llm}-missing-output.csv', index=False)
                            add_mean_row(f'{output_dir}/{version}-{llm}-missing-output.csv')

                        elif csv.startswith('means') and csv.endswith('statements_diff.csv'):
                            # print(csv)
                            df = pd.read_csv(os.path.join(version_number, file, data, csv))
                            df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
                            combined_statements.append(df)  # adiciona os novos items ao Final da lista
                            combined_df = pd.concat(combined_statements, ignore_index=True)  # concatena a lista
                            combined_df.to_csv(f'{output_dir}/{version}-{llm}-statements-output.csv', index=False)
                            add_mean_row(f'{output_dir}/{version}-{llm}-statements-output.csv')

                        elif csv.startswith('means') and csv.endswith('LINE_COVERED.csv'):
                            df = pd.read_csv(os.path.join(version_number, file, data, csv))
                            df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
                            combined_line_covered.append(df)  # adiciona os novos items ao Final da lista
                            combined_df = pd.concat(combined_line_covered, ignore_index=True)  # concatena a lista
                            combined_df.to_csv(f'{output_dir}/{version}-{llm}-LINE_COVERED-output.csv', index=False)
                            add_mean_row(f'{output_dir}/{version}-{llm}-LINE_COVERED-output.csv')

                        elif csv.startswith('means') and csv.endswith('LINE_MISSED.csv'):
                            df = pd.read_csv(os.path.join(version_number, file, data, csv))
                            df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
                            combined_line_missed.append(df)  # adiciona os novos items ao Final da lista
                            combined_df = pd.concat(combined_line_missed, ignore_index=True)  # concatena a lista
                            combined_df.to_csv(f'{output_dir}/{version}-{llm}-LINE_MISSED-output.csv', index=False)
                            add_mean_row(f'{output_dir}/{version}-{llm}-LINE_MISSED-output.csv')



# import pandas as pd
# from _dataTreatment.models.curingas import list_folders_to_menu, get_split_name
# import os
#
# ROOT = os.getcwd()
# parent_dir = os.path.dirname(ROOT)
#
# results_path = 'coverage/data_analysis/_total'
#
#
# def add_mean_row(input_csv):
#     df = pd.read_csv(input_csv)
#
#     means   = df.mean(numeric_only=True)
#     sums    = df.sum(numeric_only=True)
#
#     means['project_name'] = 'média'
#     sums['project_name'] = 'sum'
#
#     # Adicionar a linha de médias ao DataFrame
#     df_means = pd.DataFrame(means).transpose()
#     df_sums = pd.DataFrame(sums).transpose()
#     df = pd.concat([df, df_means, df_sums], ignore_index=True)
#     df.to_csv(input_csv, index=False)
#
#
# if __name__ == "__main__":
#
#     language        = list_folders_to_menu(parent_dir, word='Language')
#     data_base       = list_folders_to_menu(os.path.join(language, results_path), word='DataBase')
#     approach        = list_folders_to_menu(data_base, word='Approach')
#     llm_name        = list_folders_to_menu(approach, word='LLM')
#     version_number  = list_folders_to_menu(llm_name, word='Version')
#
#     llm         = get_split_name(llm_name, '/', -1)
#     version     = get_split_name(version_number, '/', -1)
#     language    = get_split_name(language, '/', -1)
#
#     combined_missing        = []
#     combined_statements     = []
#     combined_line_missed    = []
#     combined_line_covered   = []
#
#     output_dir = f'{parent_dir}/{language}/coverage/results/{llm}'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     files = os.listdir(version_number)
#     for file in files:
#
#         if file != 'bySmell':
#             df = pd.DataFrame(columns=['Projetc'])
#             aggregated = os.listdir(os.path.join(version_number, file))
#             for data in aggregated:
#                 if os.path.isdir(os.path.join(version_number, file, data)):
#                     for csv in os.listdir(os.path.join(version_number, file, data)):
#                         # print(csv)
#                         if csv.startswith('means') and csv.endswith('missing_diff.csv'):
#                             df = pd.read_csv(os.path.join(version_number, file, data, csv))
#                             df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
#                             combined_missing.append(df)  # adiciona os novos items ao Final da lista
#                             combined_df = pd.concat(combined_missing, ignore_index=True)  # concatena a lista
#                             combined_df.to_csv(f'{output_dir}/{version}-{llm}-missing-output.csv', index=False)
#                             add_mean_row(f'{output_dir}/{version}-{llm}-missing-output.csv')
#
#                         elif csv.startswith('means') and csv.endswith('statements_diff.csv'):
#                             df = pd.read_csv(os.path.join(version_number, file, data, csv))
#                             df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
#                             combined_statements.append(df)  # adiciona os novos items ao Final da lista
#                             combined_df = pd.concat(combined_statements, ignore_index=True)  # concatena a lista
#                             combined_df.to_csv(f'{output_dir}/{version}-{llm}-statements-output.csv', index=False)
#                             add_mean_row(f'{output_dir}/{version}-{llm}-statements-output.csv')
#
#                         # elif csv.startswith('means') and csv.endswith('LINE_COVERED.csv'):
#                         #     df = pd.read_csv(os.path.join(version_number, file, data, csv))
#                         #     df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
#                         #     combined_line_covered.append(df)  # adiciona os novos items ao Final da lista
#                         #     combined_df = pd.concat(combined_line_covered, ignore_index=True)  # concatena a lista
#                         #     combined_df.to_csv(f'{output_dir}/{version}-{llm}-LINE_COVERED-output.csv', index=False)
#                         #     add_mean_row(f'{output_dir}/{version}-{llm}-LINE_COVERED-output.csv')
#                         #
#                         # elif csv.startswith('means') and csv.endswith('LINE_MISSED.csv'):
#                         #     df = pd.read_csv(os.path.join(version_number, file, data, csv))
#                         #     df.insert(0, 'project_name', file)  # cria primeira coluna com o nome do projeto
#                         #     combined_line_missed.append(df)  # adiciona os novos items ao Final da lista
#                         #     combined_df = pd.concat(combined_line_missed, ignore_index=True)  # concatena a lista
#                         #     combined_df.to_csv(f'{output_dir}/{version}-{llm}-LINE_MISSED-output.csv', index=False)
#                         #     add_mean_row(f'{output_dir}/{version}-{llm}-LINE_MISSED-output.csv')
