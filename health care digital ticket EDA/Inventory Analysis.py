#%%
'''
Beltmann data api conn

Set up the data connection with smartsheet
'''
xMytoken='a0rcziibs8bzzchmpqwgyusd4n' 
SHEET_ID='8485507110856580'
# Smartsheet Token
xSheet = smartsheet.Smartsheet(xMytoken)


tt = xSheet.Sheets.get_sheet(SHEET_ID)
def simple_sheet_to_dataframe(sheet):
    col_names = [col.title for col in sheet.columns]
    rows = []
    for row in sheet.rows:
        cells = []
        for cell in row.cells:
            cells.append(cell.value)
        rows.append(cells)
    data_frame = pd.DataFrame(rows, columns=col_names)
    return data_frame
sm_data=simple_sheet_to_dataframe(tt)
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "give out analysis"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

datt_only=datt[:10]
path='C:/Users/Mr.Goldss/Desktop/beltmann inventory(smartsheet)/data/'
filename='MARICOPA COUNTY COMPLETE INVENTORY TRANSACTIONS'
need_col=['PROJECT','CATEGORY','ITEM/ SKU','DESCRIPTION','SOURCE','SHIPPER'
,'CASE COUNT','# INNER CASE CTN',	'# PER INNER CTN','TOTAL QTY',
'RECEIVED DATE',	'BALANCE',	'CUSTOMER NOTES','qty out','full pallet out',	'RELEASE DATE','Workorder Nbr','Release to cust:']
df_smart=sm_data
df_smart=df_smart[need_col]

#%%
'''create inventory table'''
df_smart['BALANCE']=df_smart['BALANCE'].fillna(0)
df_smart['BALANCE']=df_smart['BALANCE'].replace({'#UNPARSEABLE':0})

df_smart['BALANCE'].astype(int)
df_inv=df_smart[df_smart['BALANCE']>0]
df_inv['PROJECT']=df_inv['PROJECT'].fillna(method='ffill')
#%%
df_inv['BALANCE']=abs(df_inv['BALANCE'])
# %%
dddd=df_inv.groupby('PROJECT')['BALANCE'].sum()
#%%
dddd.plot.pie( figsize=(10,10))
# %%
plt.pie(Tasks,labels=my_labels,autopct='%1.1f%%')
plt.title('My Tasks')
plt.axis('equal')
plt.show()