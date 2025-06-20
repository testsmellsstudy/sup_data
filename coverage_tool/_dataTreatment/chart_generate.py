import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from _dataTreatment.models.curingas import list_folders_to_menu
from datetime import datetime


ROOT = os.getcwd()
parent_dir = os.path.dirname(ROOT)


def plot_stacked_bar_chart(statements_file, metric, language, directory):

    df = pd.read_csv(statements_file)

    smells_name = []
    # Set x axis label
    x_labels = df['file_name']
    for smell in x_labels:
        smells_name.append(smell.split('_')[0])

    x = np.arange(len(smells_name))
    bar_width = 0.8

    positive_values = df.loc[:, [f'sum_{metric}_GPT', f'sum_{metric}_GEMINI', f'sum_{metric}_LLAMA']].clip(lower=0)
    negative_values = df.loc[:, [f'sum_{metric}_GPT', f'sum_{metric}_GEMINI', f'sum_{metric}_LLAMA']].clip(upper=0)

    # Create chart
    plt.figure(figsize=(14, 8))

    # Plotting chart bars with positive and negative columns
    bottoms_positive = np.zeros(len(x))  # Base for stacking positive values
    bottoms_negative = np.zeros(len(x))  # Base for stacking negative values

    for i, col in enumerate(positive_values.columns):
        plt.bar(x, positive_values[col], width=bar_width, bottom=bottoms_positive, label=f"{col.split('_')[-1]}", color=f"C{i}")
        bottoms_positive += positive_values[col]  # Atualizar base para valores positivos

    for i, col in enumerate(negative_values.columns):
        plt.bar(x, negative_values[col], width=bar_width, bottom=bottoms_negative)
        bottoms_negative += negative_values[col]  # Atualizar base para valores negativos

    # plt.xlabel('Test Smells', fontsize=16)
    if metric == 'missing':
        plt.ylabel(f'Statements {metric}', fontsize=16, weight='bold')
    else:
        plt.ylabel(f'{metric}', fontsize=16, weight='bold')

    plt.xticks(x, smells_name, rotation=45, ha='right', fontsize=12, weight='bold')
    plt.yticks(weight='bold')

    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)

    plt.legend(loc='upper left', prop={'weight': 'bold'})


    # Set labels and legend
    # plt.xticks(x, smells_name, rotation=45, ha='right')
    # # plt.yticks(fontsize=19)
    # # plt.xlabel('Test Smells')
    # if metric == 'missing':
    #     plt.ylabel(f'Statements {metric}')
    # else:
    #     plt.ylabel(f'{metric}')
    # # plt.title(f'Comparative analysis of the impact of refactoring on {language} code coverage - {metric} Metric')
    # # Análise comparativa do impacto da refatoração na cobertura de código
    # plt.axhline(0, color='black', linewidth=0.8)  # Linha do eixo zero
    # # plt.legend(loc='upper left', fontsize=16, fontweight='bold')
    #
    # plt.legend(loc='upper left', prop={'fontsize': 12, 'weight': 'bold'})  # <- aqui o segredo
    output_image = f'{directory}/{language}/{metric}_stacked_bar_chart_{datetime.now()}.png'
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()

def merge_csv_files():

    # Read CSV files
    df_gpt = pd.read_csv(f'{parent_dir}/{language}/coverage/results/GPT4/{language}-GPT4_consolidated-results-bySmell.csv')
    df_gemini = pd.read_csv(f'{parent_dir}/{language}/coverage/results/GEMINI/{language}-GEMINI_consolidated-results-bySmell.csv')
    df_llama = pd.read_csv(f'{parent_dir}/{language}/coverage/results/LLAMA/{language}-LLAMA_consolidated-results-bySmell.csv')

    # Merge files based on column 'file_name'
    merged_df = df_gpt.merge(df_gemini, on='file_name', suffixes=('_GPT', '_GEMINI'))
    merged_df = merged_df.merge(df_llama, on='file_name')
    merged_df.rename(columns={
        'sum_statements': 'sum_statements_LLAMA',
        'sum_missing': 'sum_missing_LLAMA'
    }, inplace=True)

    if not os.path.exists(f'{directory}/{language}'):
        os.makedirs(f'{directory}/{language}')

    statements_df = merged_df[['file_name',
                                'sum_statements_GPT',
                                'sum_statements_GEMINI',
                                'sum_statements_LLAMA']]
    statements_df.to_csv(f'{directory}/{language}/{output_statements}', index=False)
    plot_stacked_bar_chart(f'{directory}/{language}/{output_statements}', metric='statements', language=language,
                           directory=directory)

    # For missing
    missing_df = merged_df[['file_name',
                             'sum_missing_GPT',
                             'sum_missing_GEMINI',
                             'sum_missing_LLAMA']]
    missing_df.to_csv(f'{directory}/{language}/{output_missing}', index=False)
    plot_stacked_bar_chart(f'{directory}/{language}/{output_missing}', metric='missing', language=language,
                           directory=directory)


if __name__ == '__main__':
    language = list_folders_to_menu(parent_dir, word='Language')
    language = language.split('/')[-1]
    directory = '../_analysis'

    output_statements = f"{language}_output_statements.csv"
    output_missing = f"{language}_output_missing.csv"
    merge_csv_files()