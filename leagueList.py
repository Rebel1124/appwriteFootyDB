from appwrite.client import Client
from appwrite.services.databases import Databases
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

def main(context):
    client = Client()
    
    project_id = os.environ['APPWRITE_PROJECT_ID']
    api_key = os.environ['APPWRITE_API_KEY']
    database_id = os.environ['APPWRITE_DB_ID']
    leagueList_collection_id = os.environ['LEAGUE_LIST_COLLECTION_ID']
    
    client = (client
        .set_endpoint('https://cloud.appwrite.io/v1') # Your API Endpoint
        .set_project(project_id)  # Your project ID
        .set_key(api_key) # Your secret API key
    )
    
    databases = Databases(client)
    
    #Get Leagues
    def get_league_list():
        url = "https://api.football-data-api.com/league-list"
        paramsLeague = {
            "key": "2d175fcc2f3c25f0a2a6f27e9ff2c662ca850cdc2fdac707296be94f83ddd837",
            "chosen_leagues_only": "true"
        }
        try:
            response = requests.get(url, params=paramsLeague)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            context.log(f"Error retrieving data: {e}")
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
        
        # Apply conversion to all columns except NAME columns
        for col in df.columns:
            # Skip Name columns
            if col.lower() == 'name':
                # Ensure name columns are strings
                df[col] = df[col].astype(str)
                df[col]= df.apply(lambda x: x['name'].replace(x['country'], '').replace(' ', ''), axis=1)
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
            context.log(f"Error converting value to JSON string: {str(e)}")
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
                    context.log(f"Error processing field {key}: {str(e)}")
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
            'attrName': []
        }
        
        for column in df.columns:
            # Check if column is an ID field
            # if column.lower() == 'id' or column.endswith('_id') or column.endswith('ID'):
            if column.lower() == 'name':
                column_groups['attrName'].append(column)
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
                batch_ids = [doc['name'] for doc in response['documents']]
                document_ids.extend(batch_ids)
                
                # Update the cursor for the next batch
                if batch_ids:
                    last_id = batch_ids[-1]
                else:
                    break
                    
            return document_ids
        
        except Exception as e:
            context.log(f"An error occurred: {str(e)}")
            return None
    
    def get_categorized_attributes(db_id, coll_id):
    
        categorized_attrs = {
            'string': [],
            'array': [],
            'float': [],
            'attrName': []
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
                
                
                if attr_key == 'name':
                    categorized_attrs['attrName'].append(attr_key)
                # Classify strings with null default as arrays
                elif attr_type == 'string' and default is None:
                    categorized_attrs['array'].append(attr_key)
                # Classify doubles as floats
                elif attr_type == 'double':
                    categorized_attrs['float'].append(attr_key)
                elif attr_type in categorized_attrs:
                    categorized_attrs[attr_type].append(attr_key)
                else:
                    context.log(f"Warning: Unhandled attribute type '{attr_type}' for key '{attr_key}'")
        
            return categorized_attrs
    
        except Exception as e:
            context.log(f"An error occurred: {str(e)}")
            return None
    
    
    
    def create_collection_attributes(classifications, database_id: str, collection_id: str):
        # Name attribute
        name_attribute = classifications['attrName']
        # Float attributes
        float_attributes = classifications['float']
        # String attributes
        string_attributes = classifications['string']
        # Array attributes
        array_attributes = classifications['array']
    
        try:
    
            # Create rowID attributes
            for attr in name_attribute:
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
            context.log(f"Error creating attributes: {str(e)}")
    
    # Retrieve league data
    league_data = get_league_list()
    
    if league_data and 'data' in league_data:
    
        leagueDataDF = convert_json_to_df(league_data['data'])
        leagueDataJSON=prepare_for_appwrite(leagueDataDF)
    
        docIDs = get_all_document_ids(database_id, leagueList_collection_id)
        classifications=group_columns(league_data['data'])
    
        try:
            attList = databases.list_attributes(
            database_id = database_id,
            collection_id= leagueList_collection_id 
            )
    
            attr_categories = get_categorized_attributes(database_id, leagueList_collection_id)
    
            # context.log(attList)
    
            if (attList['total'] == 0):
                # classifications=group_columns(upcoming_matches['data'])
                create_collection_attributes(
                classifications=classifications,
                database_id=database_id,
                collection_id=leagueList_collection_id 
                )
    
                context.log('Initial Attributes added for '+ leagueList_collection_id)
    
            if (len(classifications['attrName']) == len(attr_categories['attrName'])):
                context.log('name category all good')
            else:
                missingID=list(set(classifications['attrName']) - set(attr_categories['attrName']))
                # Create rowID attributes
                for attrNames in missingID:
                    databases.create_string_attribute(  # Using string for IDs instead of float
                    database_id=database_id,
                    collection_id=leagueList_collection_id,
                    key=attrNames,
                    required=True,  # IDs should be required
                    size=72000  # Adjust size based on your ID format
                    )
    
            if (len(classifications['float']) == len(attr_categories['float'])):
                context.log('float category all good')
            else:
                missingFloat=list(set(classifications['float']) - set(attr_categories['float']))
                # Create float attributes
                for attrFloats in missingFloat:
                    databases.create_float_attribute(
                    database_id=database_id,
                    collection_id=leagueList_collection_id,
                    key=attrFloats,
                    required=False,  # Set to false to allow null values
                    min=-1,  # Adjust min value based on your needs,
                    default=-1  # Default value when null
                    )
    
    
            if (len(classifications['array']) == len(attr_categories['array'])):
                context.log('array category all good')
            else:
                missingArray=list(set(classifications['array']) - set(attr_categories['array']))
                # Create array attributes
                for attrArray in missingArray:
                    databases.create_string_attribute(
                        database_id=database_id,
                        collection_id=leagueList_collection_id,
                        key=attrArray,
                        required=False,  # Set to false to allow null values
                        # array=True,
                        size=72000 # Adjust size as needed
                    )
    
    
            if (len(classifications['string']) == len(attr_categories['string'])):
                context.log('string category all good')
            else:
                missingString=list(set(classifications['string']) - set(attr_categories['string']))
        
                # Create string attributes
                for attrString in missingString:
                    databases.create_string_attribute(
                        database_id=database_id,
                        collection_id=leagueList_collection_id,
                        key=attrString,
                        required=False,  # Set to false to allow null values
                        default="",  # Default empty string when null
                        size=72000  # Adjust size as needed
                )
    
        except:
             context.log('Check attributes for '+ str(leagueList_collection_id))
    
        try:
            if(len(docIDs) > 0):
                for id in docIDs:
    
                    docUpdate=databases.delete_document(
                        database_id=database_id,
                        collection_id=leagueList_collection_id,
                        document_id=id,
                    )
                    
                    context.log(id+' Deleted')
            else:
                context.log("No matches to delete")
        except:
            context.log("No matches to delete")
                    
        for league in leagueDataJSON:
            # context.log(league['name'])
    
            try:
                databases.create_document(
                database_id=database_id,
                collection_id=leagueList_collection_id,
                document_id=league['name'],
                data=league
                )
    
                context.log('Documents created for '+league['name'])
            except Exception as e:
            
                context.log(f"\nError creating document:")
                context.log(f"Error message: {str(e)}")
                # context.log details of the problematic field
                field_name = str(e).split("'")[1].split("'")[0] if "'" in str(e) else None
                if field_name and field_name in league:
                    context.log(f"\nProblem field details:")
                    context.log(f"{field_name} type: {type(league[field_name])}")
                    context.log(f"{field_name} length: {len(str(league[field_name]))}")
                    context.log(f"Preview: {str(league[field_name])[:100]}...")
    
    return context.res.empty()


