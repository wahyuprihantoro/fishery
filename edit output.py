import pandas as pd

output = pd.read_csv('output.csv')

output['image'] = 'test_stg2/' + output['image']

output.to_csv('output_baru.csv')