from appwrite.client import Client
from appwrite.services.databases import Databases
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any
from datetime import timedelta
from datetime import datetime
from appwrite.query import Query
import json
import ast
import os

load_dotenv

project_id = os.getenv('APPWRITE_PROJECT_ID')
api_key = os.getenv('APPWRITE_API_KEY')
database_id = os.getenv('APPWRITE_DB_ID')
over_collection_id = os.getenv('OVER_LIST_COLLECTION_ID')


client = Client()

client = (client
    .set_endpoint('https://cloud.appwrite.io/v1') # Your API Endpoint
    .set_project(project_id)  # Your project ID
    .set_key(api_key) # Your secret API key
)

databases = Databases(client)

#Get over25 Stats
def get_over25_stats():
    url = "https://api.football-data-api.com/stats-data-over25"
    params = {
    "key": "2d175fcc2f3c25f0a2a6f27e9ff2c662ca850cdc2fdac707296be94f83ddd837"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error retrieving data: {e}")
        return None
    


def convert_json_to_df(json_data: Union[Dict, List]) -> pd.DataFrame:
    """
    Converts JSON data into a single DataFrame, preserving arrays/lists and converting 
    string numbers to floats, while keeping ID columns as strings and handling NaN values.
    
    Args:
        json_data (Union[Dict, List]): Either a dictionary containing nested data
                                      or a list of dictionaries
        
    Returns:
        pd.DataFrame: Single DataFrame with arrays preserved and numbers converted
    """
    # Convert to DataFrame first
    df = pd.DataFrame(json_data) if isinstance(json_data, list) else pd.DataFrame([json_data])
    
    # Function to convert string numbers to float
    def convert_to_float(val):
        if isinstance(val, str):
            try:
                return float(val)
            except (ValueError, TypeError):
                return val
        return val
    
    # Apply conversion to all columns except id columns
    for col in df.columns:
        # Skip id columns
        if col.lower() == 'id':
            # Ensure id columns are strings
            df[col] = df[col].astype(str)
            continue
            
        if df[col].dtype == 'object':  # Only process non-numeric columns
            # Check if column contains string numbers
            if df[col].apply(lambda x: isinstance(x, str) and x.replace('.', '').isdigit()).any():
                df[col] = df[col].apply(convert_to_float)
    
    # Replace NaN values with appropriate empty values based on column type
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(-1)  # Replace NaN with -1 for numeric columns
        else:
            df[col] = df[col].fillna("")  # Replace NaN with empty string for other columns
            
    return df


def should_be_string(value):
    """
    Check if a value should be converted to a string based on its content/type.
    """
    if isinstance(value, (list, dict)):
        return True
    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
        return True
    return False

def convert_to_json_string(value):
    """
    Convert a value to a proper JSON string.
    """
    try:
        if isinstance(value, str):
            # If it's a string, try to parse it first then convert to JSON string
            try:
                parsed = json.loads(value.replace("'", '"'))
                return json.dumps(parsed)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(value)
                    return json.dumps(parsed)
                except:
                    return value
        else:
            # If it's not a string, convert directly to JSON string
            return json.dumps(value)
    except Exception as e:
        print(f"Error converting value to JSON string: {str(e)}")
        return str(value)

def prepare_for_appwrite(df):
    """
    Convert a dataframe to a format suitable for Appwrite document creation.
    """
    # Convert dataframe to records
    records = df.to_dict(orient='records')
    
    # Process each record
    processed_records = []
    for record in records:
        processed_record = {}
        for key, value in record.items():
            try:
                if should_be_string(value):
                    processed_record[key] = convert_to_json_string(value)
                else:
                    processed_record[key] = value
            except Exception as e:
                print(f"Error processing field {key}: {str(e)}")
                processed_record[key] = str(value)
        
        processed_records.append(processed_record)
    
    return processed_records



def group_columns(json_list: List[Dict]) -> Dict[str, List[str]]:
    """
    Converts a list of JSON dictionaries to a DataFrame and groups columns by type.
    
    Args:
        json_list (List[Dict]): List of dictionaries containing the data
        
    Returns:
        Dict[str, List[str]]: Dictionary with 'string', 'float', and 'array' keys, 
                             each containing a list of column names of that type
    """
    # Convert JSON list to DataFrame
    
    df = convert_json_to_df(json_list)

    column_groups = {
        'string': [],
        'float': [],
        'array': [],
        'attrID': []
    }
    
    for column in df.columns:
        # Check if column is an ID field
        # if column.lower() == 'id' or column.endswith('_id') or column.endswith('ID'):
        if column.lower() == 'id':
            column_groups['attrID'].append(column)
        # Check if column contains any list
        elif df[column].apply(lambda x: isinstance(x, (list, tuple, set))).any():
            column_groups['array'].append(column)
        # Check if column can be numeric
        elif pd.to_numeric(df[column], errors='coerce').notna().any():
            column_groups['float'].append(column)
        else:
            column_groups['string'].append(column)
    
    return column_groups


def get_all_document_ids(db_id, db_collection):
    document_ids = []
    last_id = None
    limit = 100  # You can adjust this based on your needs
    
    try:
        while True:
            # Use cursor-based pagination instead of offset
            queries = [Query.limit(limit)]
            if last_id:
                queries.append(Query.cursor_after(last_id))
            
            response = databases.list_documents(
                database_id=db_id,
                collection_id=db_collection,
                queries=queries
            )
            
            if not response['documents']:
                break
                
            # Use list comprehension for better performance
            batch_ids = [doc['id'] for doc in response['documents']]
            document_ids.extend(batch_ids)
            
            # Update the cursor for the next batch
            if batch_ids:
                last_id = batch_ids[-1]
            else:
                break
                
        return document_ids
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def get_categorized_attributes(db_id, coll_id):

    categorized_attrs = {
        'string': [],
        'array': [],
        'float': [],
        'attrID': []
    }
    
    try:
        collection = databases.get_collection(
            database_id=db_id,
            collection_id=coll_id
        )
        
        # Categorize each attribute by type
        for attr in collection['attributes']:
            attr_type = attr['type']
            attr_key = attr['key']
            default = attr.get('default', None)  # Get default value if it exists
            
            
            if attr_key == 'id':
                categorized_attrs['attrID'].append(attr_key)
            # Classify strings with null default as arrays
            elif attr_type == 'string' and default is None:
                categorized_attrs['array'].append(attr_key)
            # Classify doubles as floats
            elif attr_type == 'double':
                categorized_attrs['float'].append(attr_key)
            elif attr_type in categorized_attrs:
                categorized_attrs[attr_type].append(attr_key)
            else:
                print(f"Warning: Unhandled attribute type '{attr_type}' for key '{attr_key}'")
    
        return categorized_attrs

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None



def create_collection_attributes(classifications, database_id: str, collection_id: str):
    # ID attribute
    id_attribute = classifications['attrID']
    # Float attributes
    float_attributes = classifications['float']
    # String attributes
    string_attributes = classifications['string']
    # Array attributes
    array_attributes = classifications['array']

    try:

        # Create rowID attributes
        for attr in id_attribute:
            databases.create_string_attribute(  # Using string for IDs instead of float
            database_id=database_id,
            collection_id=collection_id,
            key=attr,
            required=True,  # IDs should be required
            size=72000  # Adjust size based on your ID format
            )
            

        # Create float attributes
        for attr in float_attributes:
            databases.create_float_attribute(
            database_id=database_id,
            collection_id=collection_id,
            key=attr,
            required=False,  # Set to false to allow null values
            min=-1,  # Adjust min value based on your needs,
            default=-1  # Default value when null
            )
        
        # Create string attributes
        for attr in string_attributes:
            databases.create_string_attribute(
                database_id=database_id,
                collection_id=collection_id,
                key=attr,
                required=False,  # Set to false to allow null values
                default="",  # Default empty string when null
                size=72000  # Adjust size as needed
            )

        # Create array attributes
        for attr in array_attributes:
            databases.create_string_attribute(
                database_id=database_id,
                collection_id=collection_id,
                key=attr,
                required=False,  # Set to false to allow null values
                # array=True,
                size=72000 # Adjust size as needed
            )


    except Exception as e:
        print(f"Error creating attributes: {str(e)}")

# Retrieve over25 stats
over25_dataRaw = get_over25_stats()

if over25_dataRaw and 'data' in over25_dataRaw:

    over25_data = over25_dataRaw['data']['top_fixtures']

    if over25_data and 'data' in over25_data:

        over25DataDF = convert_json_to_df(over25_data['data'])
        over25DataJSON=prepare_for_appwrite(over25DataDF)

        docIDs = get_all_document_ids(database_id, over_collection_id)
        classifications=group_columns(over25_data['data'])

        try:
            attList = databases.list_attributes(
            database_id = database_id,
            collection_id=over_collection_id
            )

            attr_categories = get_categorized_attributes(database_id, over_collection_id)

            # print(attList)

            if (attList['total'] == 0):
                # classifications=group_columns(upcoming_matches['data'])
                create_collection_attributes(
                classifications=classifications,
                database_id=database_id,
                collection_id=over_collection_id
                )

                print('Initial Attributes added for '+ over_collection_id)

            if (len(classifications['attrID']) == len(attr_categories['attrID'])):
                print('ids category all good')
            else:
                missingID=list(set(classifications['attrID']) - set(attr_categories['attrID']))
                # Create rowID attributes
                for attrID in missingID:
                    databases.create_string_attribute(  # Using string for IDs instead of float
                    database_id=database_id,
                    collection_id=over_collection_id,
                    key=attrID,
                    required=True,  # IDs should be required
                    size=72000  # Adjust size based on your ID format
                    )

            if (len(classifications['float']) == len(attr_categories['float'])):
                print('float category all good')
            else:
                missingFloat=list(set(classifications['float']) - set(attr_categories['float']))
                # Create float attributes
                for attrFloats in missingFloat:
                    databases.create_float_attribute(
                    database_id=database_id,
                    collection_id=over_collection_id,
                    key=attrFloats,
                    required=False,  # Set to false to allow null values
                    min=-1,  # Adjust min value based on your needs,
                    default=-1  # Default value when null
                    )


            if (len(classifications['array']) == len(attr_categories['array'])):
                print('array category all good')
            else:
                missingArray=list(set(classifications['array']) - set(attr_categories['array']))
                # Create array attributes
                for attrArray in missingArray:
                    databases.create_string_attribute(
                        database_id=database_id,
                        collection_id=over_collection_id,
                        key=attrArray,
                        required=False,  # Set to false to allow null values
                        # array=True,
                        size=72000 # Adjust size as needed
                    )


            if (len(classifications['string']) == len(attr_categories['string'])):
                print('string category all good')
            else:
                missingString=list(set(classifications['string']) - set(attr_categories['string']))
        
                # Create string attributes
                for attrString in missingString:
                    databases.create_string_attribute(
                        database_id=database_id,
                        collection_id=over_collection_id,
                        key=attrString,
                        required=False,  # Set to false to allow null values
                        default="",  # Default empty string when null
                        size=72000  # Adjust size as needed
                )

        except:
            print('Check attributes for '+ str(over_collection_id))

        try:
            if(len(docIDs) > 0):
                for id in docIDs:

                    docUpdate=databases.delete_document(
                        database_id=database_id,
                        collection_id=over_collection_id,
                        document_id=id,
                    )
                    
                    print(id+' Deleted')
            else:
                print("No matches to delete")
        except:
            print("No matches to delete")
                    
        for over25Fixture in over25DataJSON:

            try:
                databases.create_document(
                database_id=database_id,
                collection_id=over_collection_id,
                document_id=over25Fixture['id'],
                data=over25Fixture
                )

                print('Documents created for '+over25Fixture['id'])
            except Exception as e:
            
                print(f"\nError creating document:")
                print(f"Error message: {str(e)}")
                # Print details of the problematic field
                field_name = str(e).split("'")[1].split("'")[0] if "'" in str(e) else None
                if field_name and field_name in over25Fixture:
                    print(f"\nProblem field details:")
                    print(f"{field_name} type: {type(over25Fixture[field_name])}")
                    print(f"{field_name} length: {len(str(over25Fixture[field_name]))}")
                    print(f"Preview: {str(over25Fixture[field_name])[:100]}...")


