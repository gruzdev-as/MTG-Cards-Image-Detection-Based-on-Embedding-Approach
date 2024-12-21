import pandas as pd

from sqlalchemy import create_engine
from typing import Dict

class PG_DB_Loader:

    def __init__(self, connection_params:Dict[str:str]):
        
        connection_url = 'postgresql+psycoph2::/{USERNAME}:{PASSWORD}@{PG_IP}:{PG_PORT}/{DB_NAME}'
        self.engine = create_engine(connection_url.format(connection_params))

    def upload_recognized_data(self, card_df:pd.DataFrame) -> None:

        try: 
            card_df.to_sql('user_cards', self.engine, if_exists='append', index=False)
            print('Data uploaded successfully!')
        except Exception as e: 
            print(f'Error: {e}')
    
    def load_data_from_table(self, table_name:str, columns:list=[]) -> pd.DataFrame: 
        
        if len(columns) == 0:
            query = f'select * from {table_name}'
        else:
            query = f'select {','.join(columns)} from {table_name}'
        try: 
            df = pd.read_sql(query, self.engine)
            print('Loaded successfully')
        except Exception as e:
            print(f'Error: {e}')
        
        return df