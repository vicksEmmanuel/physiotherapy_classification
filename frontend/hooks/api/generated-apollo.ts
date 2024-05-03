import { gql } from '@apollo/client';
import * as Apollo from '@apollo/client';
export type Maybe<T> = T | null;
export type InputMaybe<T> = Maybe<T>;
export type Exact<T extends { [key: string]: unknown }> = { [K in keyof T]: T[K] };
export type MakeOptional<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]?: Maybe<T[SubKey]> };
export type MakeMaybe<T, K extends keyof T> = Omit<T, K> & { [SubKey in K]: Maybe<T[SubKey]> };
const defaultOptions = {} as const;
/** All built-in and custom scalars, mapped to their actual values */
export type Scalars = {
  ID: string;
  String: string;
  Boolean: boolean;
  Int: number;
  Float: number;
  Any: any;
  timestamptz: any;
  uuid: any;
};

export type AddUserTokenPayload = {
  token: Scalars['String'];
  userId: Scalars['String'];
};

export type AddUserTokenResponse = {
  __typename?: 'AddUserTokenResponse';
  smart_sheet?: Maybe<Scalars['Any']>;
  token: Scalars['String'];
};

/** Boolean expression to compare columns of type "Boolean". All fields are combined with logical 'AND'. */
export type Boolean_Comparison_Exp = {
  _eq?: InputMaybe<Scalars['Boolean']>;
  _gt?: InputMaybe<Scalars['Boolean']>;
  _gte?: InputMaybe<Scalars['Boolean']>;
  _in?: InputMaybe<Array<Scalars['Boolean']>>;
  _is_null?: InputMaybe<Scalars['Boolean']>;
  _lt?: InputMaybe<Scalars['Boolean']>;
  _lte?: InputMaybe<Scalars['Boolean']>;
  _neq?: InputMaybe<Scalars['Boolean']>;
  _nin?: InputMaybe<Array<Scalars['Boolean']>>;
};

/** Boolean expression to compare columns of type "Int". All fields are combined with logical 'AND'. */
export type Int_Comparison_Exp = {
  _eq?: InputMaybe<Scalars['Int']>;
  _gt?: InputMaybe<Scalars['Int']>;
  _gte?: InputMaybe<Scalars['Int']>;
  _in?: InputMaybe<Array<Scalars['Int']>>;
  _is_null?: InputMaybe<Scalars['Boolean']>;
  _lt?: InputMaybe<Scalars['Int']>;
  _lte?: InputMaybe<Scalars['Int']>;
  _neq?: InputMaybe<Scalars['Int']>;
  _nin?: InputMaybe<Array<Scalars['Int']>>;
};

/** Boolean expression to compare columns of type "String". All fields are combined with logical 'AND'. */
export type String_Comparison_Exp = {
  _eq?: InputMaybe<Scalars['String']>;
  _gt?: InputMaybe<Scalars['String']>;
  _gte?: InputMaybe<Scalars['String']>;
  /** does the column match the given case-insensitive pattern */
  _ilike?: InputMaybe<Scalars['String']>;
  _in?: InputMaybe<Array<Scalars['String']>>;
  /** does the column match the given POSIX regular expression, case insensitive */
  _iregex?: InputMaybe<Scalars['String']>;
  _is_null?: InputMaybe<Scalars['Boolean']>;
  /** does the column match the given pattern */
  _like?: InputMaybe<Scalars['String']>;
  _lt?: InputMaybe<Scalars['String']>;
  _lte?: InputMaybe<Scalars['String']>;
  _neq?: InputMaybe<Scalars['String']>;
  /** does the column NOT match the given case-insensitive pattern */
  _nilike?: InputMaybe<Scalars['String']>;
  _nin?: InputMaybe<Array<Scalars['String']>>;
  /** does the column NOT match the given POSIX regular expression, case insensitive */
  _niregex?: InputMaybe<Scalars['String']>;
  /** does the column NOT match the given pattern */
  _nlike?: InputMaybe<Scalars['String']>;
  /** does the column NOT match the given POSIX regular expression, case sensitive */
  _nregex?: InputMaybe<Scalars['String']>;
  /** does the column NOT match the given SQL regular expression */
  _nsimilar?: InputMaybe<Scalars['String']>;
  /** does the column match the given POSIX regular expression, case sensitive */
  _regex?: InputMaybe<Scalars['String']>;
  /** does the column match the given SQL regular expression */
  _similar?: InputMaybe<Scalars['String']>;
};

/** ordering argument of a cursor */
export enum Cursor_Ordering {
  /** ascending ordering of the cursor */
  Asc = 'ASC',
  /** descending ordering of the cursor */
  Desc = 'DESC'
}

/** columns and relationships of "frequency" */
export type Frequency = {
  __typename?: 'frequency';
  currency?: Maybe<Scalars['String']>;
  description: Scalars['String'];
  montly_price: Scalars['String'];
  /** An array relationship */
  smartsheets: Array<Smartsheet>;
  /** An aggregate relationship */
  smartsheets_aggregate: Smartsheet_Aggregate;
  title: Scalars['String'];
  yearly_price: Scalars['String'];
};


/** columns and relationships of "frequency" */
export type FrequencySmartsheetsArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


/** columns and relationships of "frequency" */
export type FrequencySmartsheets_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};

/** aggregated selection of "frequency" */
export type Frequency_Aggregate = {
  __typename?: 'frequency_aggregate';
  aggregate?: Maybe<Frequency_Aggregate_Fields>;
  nodes: Array<Frequency>;
};

/** aggregate fields of "frequency" */
export type Frequency_Aggregate_Fields = {
  __typename?: 'frequency_aggregate_fields';
  count: Scalars['Int'];
  max?: Maybe<Frequency_Max_Fields>;
  min?: Maybe<Frequency_Min_Fields>;
};


/** aggregate fields of "frequency" */
export type Frequency_Aggregate_FieldsCountArgs = {
  columns?: InputMaybe<Array<Frequency_Select_Column>>;
  distinct?: InputMaybe<Scalars['Boolean']>;
};

/** Boolean expression to filter rows from the table "frequency". All fields are combined with a logical 'AND'. */
export type Frequency_Bool_Exp = {
  _and?: InputMaybe<Array<Frequency_Bool_Exp>>;
  _not?: InputMaybe<Frequency_Bool_Exp>;
  _or?: InputMaybe<Array<Frequency_Bool_Exp>>;
  currency?: InputMaybe<String_Comparison_Exp>;
  description?: InputMaybe<String_Comparison_Exp>;
  montly_price?: InputMaybe<String_Comparison_Exp>;
  smartsheets?: InputMaybe<Smartsheet_Bool_Exp>;
  smartsheets_aggregate?: InputMaybe<Smartsheet_Aggregate_Bool_Exp>;
  title?: InputMaybe<String_Comparison_Exp>;
  yearly_price?: InputMaybe<String_Comparison_Exp>;
};

/** unique or primary key constraints on table "frequency" */
export enum Frequency_Constraint {
  /** unique or primary key constraint on columns "title" */
  FrequencyPkey = 'frequency_pkey'
}

/** input type for inserting data into table "frequency" */
export type Frequency_Insert_Input = {
  currency?: InputMaybe<Scalars['String']>;
  description?: InputMaybe<Scalars['String']>;
  montly_price?: InputMaybe<Scalars['String']>;
  smartsheets?: InputMaybe<Smartsheet_Arr_Rel_Insert_Input>;
  title?: InputMaybe<Scalars['String']>;
  yearly_price?: InputMaybe<Scalars['String']>;
};

/** aggregate max on columns */
export type Frequency_Max_Fields = {
  __typename?: 'frequency_max_fields';
  currency?: Maybe<Scalars['String']>;
  description?: Maybe<Scalars['String']>;
  montly_price?: Maybe<Scalars['String']>;
  title?: Maybe<Scalars['String']>;
  yearly_price?: Maybe<Scalars['String']>;
};

/** aggregate min on columns */
export type Frequency_Min_Fields = {
  __typename?: 'frequency_min_fields';
  currency?: Maybe<Scalars['String']>;
  description?: Maybe<Scalars['String']>;
  montly_price?: Maybe<Scalars['String']>;
  title?: Maybe<Scalars['String']>;
  yearly_price?: Maybe<Scalars['String']>;
};

/** response of any mutation on the table "frequency" */
export type Frequency_Mutation_Response = {
  __typename?: 'frequency_mutation_response';
  /** number of rows affected by the mutation */
  affected_rows: Scalars['Int'];
  /** data from the rows affected by the mutation */
  returning: Array<Frequency>;
};

/** input type for inserting object relation for remote table "frequency" */
export type Frequency_Obj_Rel_Insert_Input = {
  data: Frequency_Insert_Input;
  /** upsert condition */
  on_conflict?: InputMaybe<Frequency_On_Conflict>;
};

/** on_conflict condition type for table "frequency" */
export type Frequency_On_Conflict = {
  constraint: Frequency_Constraint;
  update_columns?: Array<Frequency_Update_Column>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};

/** Ordering options when selecting data from "frequency". */
export type Frequency_Order_By = {
  currency?: InputMaybe<Order_By>;
  description?: InputMaybe<Order_By>;
  montly_price?: InputMaybe<Order_By>;
  smartsheets_aggregate?: InputMaybe<Smartsheet_Aggregate_Order_By>;
  title?: InputMaybe<Order_By>;
  yearly_price?: InputMaybe<Order_By>;
};

/** primary key columns input for table: frequency */
export type Frequency_Pk_Columns_Input = {
  title: Scalars['String'];
};

/** select columns of table "frequency" */
export enum Frequency_Select_Column {
  /** column name */
  Currency = 'currency',
  /** column name */
  Description = 'description',
  /** column name */
  MontlyPrice = 'montly_price',
  /** column name */
  Title = 'title',
  /** column name */
  YearlyPrice = 'yearly_price'
}

/** input type for updating data in table "frequency" */
export type Frequency_Set_Input = {
  currency?: InputMaybe<Scalars['String']>;
  description?: InputMaybe<Scalars['String']>;
  montly_price?: InputMaybe<Scalars['String']>;
  title?: InputMaybe<Scalars['String']>;
  yearly_price?: InputMaybe<Scalars['String']>;
};

/** Streaming cursor of the table "frequency" */
export type Frequency_Stream_Cursor_Input = {
  /** Stream column input with initial value */
  initial_value: Frequency_Stream_Cursor_Value_Input;
  /** cursor ordering */
  ordering?: InputMaybe<Cursor_Ordering>;
};

/** Initial value of the column from where the streaming should start */
export type Frequency_Stream_Cursor_Value_Input = {
  currency?: InputMaybe<Scalars['String']>;
  description?: InputMaybe<Scalars['String']>;
  montly_price?: InputMaybe<Scalars['String']>;
  title?: InputMaybe<Scalars['String']>;
  yearly_price?: InputMaybe<Scalars['String']>;
};

/** update columns of table "frequency" */
export enum Frequency_Update_Column {
  /** column name */
  Currency = 'currency',
  /** column name */
  Description = 'description',
  /** column name */
  MontlyPrice = 'montly_price',
  /** column name */
  Title = 'title',
  /** column name */
  YearlyPrice = 'yearly_price'
}

export type Frequency_Updates = {
  /** sets the columns of the filtered rows to the given values */
  _set?: InputMaybe<Frequency_Set_Input>;
  /** filter the rows which have to be updated */
  where: Frequency_Bool_Exp;
};

/** mutation root */
export type Mutation_Root = {
  __typename?: 'mutation_root';
  addUserToken?: Maybe<AddUserTokenResponse>;
  /** delete data from the table: "frequency" */
  delete_frequency?: Maybe<Frequency_Mutation_Response>;
  /** delete single row from the table: "frequency" */
  delete_frequency_by_pk?: Maybe<Frequency>;
  /** delete data from the table: "smartsheet" */
  delete_smartsheet?: Maybe<Smartsheet_Mutation_Response>;
  /** delete single row from the table: "smartsheet" */
  delete_smartsheet_by_pk?: Maybe<Smartsheet>;
  /** delete data from the table: "user" */
  delete_user?: Maybe<User_Mutation_Response>;
  /** delete single row from the table: "user" */
  delete_user_by_pk?: Maybe<User>;
  /** insert data into the table: "frequency" */
  insert_frequency?: Maybe<Frequency_Mutation_Response>;
  /** insert a single row into the table: "frequency" */
  insert_frequency_one?: Maybe<Frequency>;
  /** insert data into the table: "smartsheet" */
  insert_smartsheet?: Maybe<Smartsheet_Mutation_Response>;
  /** insert a single row into the table: "smartsheet" */
  insert_smartsheet_one?: Maybe<Smartsheet>;
  /** insert data into the table: "user" */
  insert_user?: Maybe<User_Mutation_Response>;
  /** insert a single row into the table: "user" */
  insert_user_one?: Maybe<User>;
  /** update data of the table: "frequency" */
  update_frequency?: Maybe<Frequency_Mutation_Response>;
  /** update single row of the table: "frequency" */
  update_frequency_by_pk?: Maybe<Frequency>;
  /** update multiples rows of table: "frequency" */
  update_frequency_many?: Maybe<Array<Maybe<Frequency_Mutation_Response>>>;
  /** update data of the table: "smartsheet" */
  update_smartsheet?: Maybe<Smartsheet_Mutation_Response>;
  /** update single row of the table: "smartsheet" */
  update_smartsheet_by_pk?: Maybe<Smartsheet>;
  /** update multiples rows of table: "smartsheet" */
  update_smartsheet_many?: Maybe<Array<Maybe<Smartsheet_Mutation_Response>>>;
  /** update data of the table: "user" */
  update_user?: Maybe<User_Mutation_Response>;
  /** update single row of the table: "user" */
  update_user_by_pk?: Maybe<User>;
  /** update multiples rows of table: "user" */
  update_user_many?: Maybe<Array<Maybe<User_Mutation_Response>>>;
};


/** mutation root */
export type Mutation_RootAddUserTokenArgs = {
  input: AddUserTokenPayload;
};


/** mutation root */
export type Mutation_RootDelete_FrequencyArgs = {
  where: Frequency_Bool_Exp;
};


/** mutation root */
export type Mutation_RootDelete_Frequency_By_PkArgs = {
  title: Scalars['String'];
};


/** mutation root */
export type Mutation_RootDelete_SmartsheetArgs = {
  where: Smartsheet_Bool_Exp;
};


/** mutation root */
export type Mutation_RootDelete_Smartsheet_By_PkArgs = {
  id: Scalars['uuid'];
};


/** mutation root */
export type Mutation_RootDelete_UserArgs = {
  where: User_Bool_Exp;
};


/** mutation root */
export type Mutation_RootDelete_User_By_PkArgs = {
  id: Scalars['uuid'];
};


/** mutation root */
export type Mutation_RootInsert_FrequencyArgs = {
  objects: Array<Frequency_Insert_Input>;
  on_conflict?: InputMaybe<Frequency_On_Conflict>;
};


/** mutation root */
export type Mutation_RootInsert_Frequency_OneArgs = {
  object: Frequency_Insert_Input;
  on_conflict?: InputMaybe<Frequency_On_Conflict>;
};


/** mutation root */
export type Mutation_RootInsert_SmartsheetArgs = {
  objects: Array<Smartsheet_Insert_Input>;
  on_conflict?: InputMaybe<Smartsheet_On_Conflict>;
};


/** mutation root */
export type Mutation_RootInsert_Smartsheet_OneArgs = {
  object: Smartsheet_Insert_Input;
  on_conflict?: InputMaybe<Smartsheet_On_Conflict>;
};


/** mutation root */
export type Mutation_RootInsert_UserArgs = {
  objects: Array<User_Insert_Input>;
  on_conflict?: InputMaybe<User_On_Conflict>;
};


/** mutation root */
export type Mutation_RootInsert_User_OneArgs = {
  object: User_Insert_Input;
  on_conflict?: InputMaybe<User_On_Conflict>;
};


/** mutation root */
export type Mutation_RootUpdate_FrequencyArgs = {
  _set?: InputMaybe<Frequency_Set_Input>;
  where: Frequency_Bool_Exp;
};


/** mutation root */
export type Mutation_RootUpdate_Frequency_By_PkArgs = {
  _set?: InputMaybe<Frequency_Set_Input>;
  pk_columns: Frequency_Pk_Columns_Input;
};


/** mutation root */
export type Mutation_RootUpdate_Frequency_ManyArgs = {
  updates: Array<Frequency_Updates>;
};


/** mutation root */
export type Mutation_RootUpdate_SmartsheetArgs = {
  _set?: InputMaybe<Smartsheet_Set_Input>;
  where: Smartsheet_Bool_Exp;
};


/** mutation root */
export type Mutation_RootUpdate_Smartsheet_By_PkArgs = {
  _set?: InputMaybe<Smartsheet_Set_Input>;
  pk_columns: Smartsheet_Pk_Columns_Input;
};


/** mutation root */
export type Mutation_RootUpdate_Smartsheet_ManyArgs = {
  updates: Array<Smartsheet_Updates>;
};


/** mutation root */
export type Mutation_RootUpdate_UserArgs = {
  _set?: InputMaybe<User_Set_Input>;
  where: User_Bool_Exp;
};


/** mutation root */
export type Mutation_RootUpdate_User_By_PkArgs = {
  _set?: InputMaybe<User_Set_Input>;
  pk_columns: User_Pk_Columns_Input;
};


/** mutation root */
export type Mutation_RootUpdate_User_ManyArgs = {
  updates: Array<User_Updates>;
};

/** column ordering options */
export enum Order_By {
  /** in ascending order, nulls last */
  Asc = 'asc',
  /** in ascending order, nulls first */
  AscNullsFirst = 'asc_nulls_first',
  /** in ascending order, nulls last */
  AscNullsLast = 'asc_nulls_last',
  /** in descending order, nulls first */
  Desc = 'desc',
  /** in descending order, nulls first */
  DescNullsFirst = 'desc_nulls_first',
  /** in descending order, nulls last */
  DescNullsLast = 'desc_nulls_last'
}

export type Query_Root = {
  __typename?: 'query_root';
  /** fetch data from the table: "frequency" */
  frequency: Array<Frequency>;
  /** fetch aggregated fields from the table: "frequency" */
  frequency_aggregate: Frequency_Aggregate;
  /** fetch data from the table: "frequency" using primary key columns */
  frequency_by_pk?: Maybe<Frequency>;
  hello?: Maybe<Scalars['String']>;
  /** fetch data from the table: "smartsheet" */
  smartsheet: Array<Smartsheet>;
  /** fetch aggregated fields from the table: "smartsheet" */
  smartsheet_aggregate: Smartsheet_Aggregate;
  /** fetch data from the table: "smartsheet" using primary key columns */
  smartsheet_by_pk?: Maybe<Smartsheet>;
  /** fetch data from the table: "user" */
  user: Array<User>;
  /** fetch aggregated fields from the table: "user" */
  user_aggregate: User_Aggregate;
  /** fetch data from the table: "user" using primary key columns */
  user_by_pk?: Maybe<User>;
};


export type Query_RootFrequencyArgs = {
  distinct_on?: InputMaybe<Array<Frequency_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Frequency_Order_By>>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};


export type Query_RootFrequency_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Frequency_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Frequency_Order_By>>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};


export type Query_RootFrequency_By_PkArgs = {
  title: Scalars['String'];
};


export type Query_RootSmartsheetArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


export type Query_RootSmartsheet_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


export type Query_RootSmartsheet_By_PkArgs = {
  id: Scalars['uuid'];
};


export type Query_RootUserArgs = {
  distinct_on?: InputMaybe<Array<User_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<User_Order_By>>;
  where?: InputMaybe<User_Bool_Exp>;
};


export type Query_RootUser_AggregateArgs = {
  distinct_on?: InputMaybe<Array<User_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<User_Order_By>>;
  where?: InputMaybe<User_Bool_Exp>;
};


export type Query_RootUser_By_PkArgs = {
  id: Scalars['uuid'];
};

/** columns and relationships of "smartsheet" */
export type Smartsheet = {
  __typename?: 'smartsheet';
  createdAt?: Maybe<Scalars['timestamptz']>;
  frequency?: Maybe<Scalars['String']>;
  /** An object relationship */
  frequencyByFrequency?: Maybe<Frequency>;
  id: Scalars['uuid'];
  is_monthly_payment?: Maybe<Scalars['Boolean']>;
  is_payment_set_up?: Maybe<Scalars['Boolean']>;
  lastSyncedAt?: Maybe<Scalars['timestamptz']>;
  rules_sheet?: Maybe<Scalars['String']>;
  smart_sheet_api_token: Scalars['String'];
  updatedAt?: Maybe<Scalars['timestamptz']>;
  /** An object relationship */
  user: User;
  userId: Scalars['uuid'];
};

/** aggregated selection of "smartsheet" */
export type Smartsheet_Aggregate = {
  __typename?: 'smartsheet_aggregate';
  aggregate?: Maybe<Smartsheet_Aggregate_Fields>;
  nodes: Array<Smartsheet>;
};

export type Smartsheet_Aggregate_Bool_Exp = {
  bool_and?: InputMaybe<Smartsheet_Aggregate_Bool_Exp_Bool_And>;
  bool_or?: InputMaybe<Smartsheet_Aggregate_Bool_Exp_Bool_Or>;
  count?: InputMaybe<Smartsheet_Aggregate_Bool_Exp_Count>;
};

export type Smartsheet_Aggregate_Bool_Exp_Bool_And = {
  arguments: Smartsheet_Select_Column_Smartsheet_Aggregate_Bool_Exp_Bool_And_Arguments_Columns;
  distinct?: InputMaybe<Scalars['Boolean']>;
  filter?: InputMaybe<Smartsheet_Bool_Exp>;
  predicate: Boolean_Comparison_Exp;
};

export type Smartsheet_Aggregate_Bool_Exp_Bool_Or = {
  arguments: Smartsheet_Select_Column_Smartsheet_Aggregate_Bool_Exp_Bool_Or_Arguments_Columns;
  distinct?: InputMaybe<Scalars['Boolean']>;
  filter?: InputMaybe<Smartsheet_Bool_Exp>;
  predicate: Boolean_Comparison_Exp;
};

export type Smartsheet_Aggregate_Bool_Exp_Count = {
  arguments?: InputMaybe<Array<Smartsheet_Select_Column>>;
  distinct?: InputMaybe<Scalars['Boolean']>;
  filter?: InputMaybe<Smartsheet_Bool_Exp>;
  predicate: Int_Comparison_Exp;
};

/** aggregate fields of "smartsheet" */
export type Smartsheet_Aggregate_Fields = {
  __typename?: 'smartsheet_aggregate_fields';
  count: Scalars['Int'];
  max?: Maybe<Smartsheet_Max_Fields>;
  min?: Maybe<Smartsheet_Min_Fields>;
};


/** aggregate fields of "smartsheet" */
export type Smartsheet_Aggregate_FieldsCountArgs = {
  columns?: InputMaybe<Array<Smartsheet_Select_Column>>;
  distinct?: InputMaybe<Scalars['Boolean']>;
};

/** order by aggregate values of table "smartsheet" */
export type Smartsheet_Aggregate_Order_By = {
  count?: InputMaybe<Order_By>;
  max?: InputMaybe<Smartsheet_Max_Order_By>;
  min?: InputMaybe<Smartsheet_Min_Order_By>;
};

/** input type for inserting array relation for remote table "smartsheet" */
export type Smartsheet_Arr_Rel_Insert_Input = {
  data: Array<Smartsheet_Insert_Input>;
  /** upsert condition */
  on_conflict?: InputMaybe<Smartsheet_On_Conflict>;
};

/** Boolean expression to filter rows from the table "smartsheet". All fields are combined with a logical 'AND'. */
export type Smartsheet_Bool_Exp = {
  _and?: InputMaybe<Array<Smartsheet_Bool_Exp>>;
  _not?: InputMaybe<Smartsheet_Bool_Exp>;
  _or?: InputMaybe<Array<Smartsheet_Bool_Exp>>;
  createdAt?: InputMaybe<Timestamptz_Comparison_Exp>;
  frequency?: InputMaybe<String_Comparison_Exp>;
  frequencyByFrequency?: InputMaybe<Frequency_Bool_Exp>;
  id?: InputMaybe<Uuid_Comparison_Exp>;
  is_monthly_payment?: InputMaybe<Boolean_Comparison_Exp>;
  is_payment_set_up?: InputMaybe<Boolean_Comparison_Exp>;
  lastSyncedAt?: InputMaybe<Timestamptz_Comparison_Exp>;
  rules_sheet?: InputMaybe<String_Comparison_Exp>;
  smart_sheet_api_token?: InputMaybe<String_Comparison_Exp>;
  updatedAt?: InputMaybe<Timestamptz_Comparison_Exp>;
  user?: InputMaybe<User_Bool_Exp>;
  userId?: InputMaybe<Uuid_Comparison_Exp>;
};

/** unique or primary key constraints on table "smartsheet" */
export enum Smartsheet_Constraint {
  /** unique or primary key constraint on columns "id" */
  SmartsheetPkey = 'smartsheet_pkey'
}

/** input type for inserting data into table "smartsheet" */
export type Smartsheet_Insert_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  frequency?: InputMaybe<Scalars['String']>;
  frequencyByFrequency?: InputMaybe<Frequency_Obj_Rel_Insert_Input>;
  id?: InputMaybe<Scalars['uuid']>;
  is_monthly_payment?: InputMaybe<Scalars['Boolean']>;
  is_payment_set_up?: InputMaybe<Scalars['Boolean']>;
  lastSyncedAt?: InputMaybe<Scalars['timestamptz']>;
  rules_sheet?: InputMaybe<Scalars['String']>;
  smart_sheet_api_token?: InputMaybe<Scalars['String']>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
  user?: InputMaybe<User_Obj_Rel_Insert_Input>;
  userId?: InputMaybe<Scalars['uuid']>;
};

/** aggregate max on columns */
export type Smartsheet_Max_Fields = {
  __typename?: 'smartsheet_max_fields';
  createdAt?: Maybe<Scalars['timestamptz']>;
  frequency?: Maybe<Scalars['String']>;
  id?: Maybe<Scalars['uuid']>;
  lastSyncedAt?: Maybe<Scalars['timestamptz']>;
  rules_sheet?: Maybe<Scalars['String']>;
  smart_sheet_api_token?: Maybe<Scalars['String']>;
  updatedAt?: Maybe<Scalars['timestamptz']>;
  userId?: Maybe<Scalars['uuid']>;
};

/** order by max() on columns of table "smartsheet" */
export type Smartsheet_Max_Order_By = {
  createdAt?: InputMaybe<Order_By>;
  frequency?: InputMaybe<Order_By>;
  id?: InputMaybe<Order_By>;
  lastSyncedAt?: InputMaybe<Order_By>;
  rules_sheet?: InputMaybe<Order_By>;
  smart_sheet_api_token?: InputMaybe<Order_By>;
  updatedAt?: InputMaybe<Order_By>;
  userId?: InputMaybe<Order_By>;
};

/** aggregate min on columns */
export type Smartsheet_Min_Fields = {
  __typename?: 'smartsheet_min_fields';
  createdAt?: Maybe<Scalars['timestamptz']>;
  frequency?: Maybe<Scalars['String']>;
  id?: Maybe<Scalars['uuid']>;
  lastSyncedAt?: Maybe<Scalars['timestamptz']>;
  rules_sheet?: Maybe<Scalars['String']>;
  smart_sheet_api_token?: Maybe<Scalars['String']>;
  updatedAt?: Maybe<Scalars['timestamptz']>;
  userId?: Maybe<Scalars['uuid']>;
};

/** order by min() on columns of table "smartsheet" */
export type Smartsheet_Min_Order_By = {
  createdAt?: InputMaybe<Order_By>;
  frequency?: InputMaybe<Order_By>;
  id?: InputMaybe<Order_By>;
  lastSyncedAt?: InputMaybe<Order_By>;
  rules_sheet?: InputMaybe<Order_By>;
  smart_sheet_api_token?: InputMaybe<Order_By>;
  updatedAt?: InputMaybe<Order_By>;
  userId?: InputMaybe<Order_By>;
};

/** response of any mutation on the table "smartsheet" */
export type Smartsheet_Mutation_Response = {
  __typename?: 'smartsheet_mutation_response';
  /** number of rows affected by the mutation */
  affected_rows: Scalars['Int'];
  /** data from the rows affected by the mutation */
  returning: Array<Smartsheet>;
};

/** on_conflict condition type for table "smartsheet" */
export type Smartsheet_On_Conflict = {
  constraint: Smartsheet_Constraint;
  update_columns?: Array<Smartsheet_Update_Column>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};

/** Ordering options when selecting data from "smartsheet". */
export type Smartsheet_Order_By = {
  createdAt?: InputMaybe<Order_By>;
  frequency?: InputMaybe<Order_By>;
  frequencyByFrequency?: InputMaybe<Frequency_Order_By>;
  id?: InputMaybe<Order_By>;
  is_monthly_payment?: InputMaybe<Order_By>;
  is_payment_set_up?: InputMaybe<Order_By>;
  lastSyncedAt?: InputMaybe<Order_By>;
  rules_sheet?: InputMaybe<Order_By>;
  smart_sheet_api_token?: InputMaybe<Order_By>;
  updatedAt?: InputMaybe<Order_By>;
  user?: InputMaybe<User_Order_By>;
  userId?: InputMaybe<Order_By>;
};

/** primary key columns input for table: smartsheet */
export type Smartsheet_Pk_Columns_Input = {
  id: Scalars['uuid'];
};

/** select columns of table "smartsheet" */
export enum Smartsheet_Select_Column {
  /** column name */
  CreatedAt = 'createdAt',
  /** column name */
  Frequency = 'frequency',
  /** column name */
  Id = 'id',
  /** column name */
  IsMonthlyPayment = 'is_monthly_payment',
  /** column name */
  IsPaymentSetUp = 'is_payment_set_up',
  /** column name */
  LastSyncedAt = 'lastSyncedAt',
  /** column name */
  RulesSheet = 'rules_sheet',
  /** column name */
  SmartSheetApiToken = 'smart_sheet_api_token',
  /** column name */
  UpdatedAt = 'updatedAt',
  /** column name */
  UserId = 'userId'
}

/** select "smartsheet_aggregate_bool_exp_bool_and_arguments_columns" columns of table "smartsheet" */
export enum Smartsheet_Select_Column_Smartsheet_Aggregate_Bool_Exp_Bool_And_Arguments_Columns {
  /** column name */
  IsMonthlyPayment = 'is_monthly_payment',
  /** column name */
  IsPaymentSetUp = 'is_payment_set_up'
}

/** select "smartsheet_aggregate_bool_exp_bool_or_arguments_columns" columns of table "smartsheet" */
export enum Smartsheet_Select_Column_Smartsheet_Aggregate_Bool_Exp_Bool_Or_Arguments_Columns {
  /** column name */
  IsMonthlyPayment = 'is_monthly_payment',
  /** column name */
  IsPaymentSetUp = 'is_payment_set_up'
}

/** input type for updating data in table "smartsheet" */
export type Smartsheet_Set_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  frequency?: InputMaybe<Scalars['String']>;
  id?: InputMaybe<Scalars['uuid']>;
  is_monthly_payment?: InputMaybe<Scalars['Boolean']>;
  is_payment_set_up?: InputMaybe<Scalars['Boolean']>;
  lastSyncedAt?: InputMaybe<Scalars['timestamptz']>;
  rules_sheet?: InputMaybe<Scalars['String']>;
  smart_sheet_api_token?: InputMaybe<Scalars['String']>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
  userId?: InputMaybe<Scalars['uuid']>;
};

/** Streaming cursor of the table "smartsheet" */
export type Smartsheet_Stream_Cursor_Input = {
  /** Stream column input with initial value */
  initial_value: Smartsheet_Stream_Cursor_Value_Input;
  /** cursor ordering */
  ordering?: InputMaybe<Cursor_Ordering>;
};

/** Initial value of the column from where the streaming should start */
export type Smartsheet_Stream_Cursor_Value_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  frequency?: InputMaybe<Scalars['String']>;
  id?: InputMaybe<Scalars['uuid']>;
  is_monthly_payment?: InputMaybe<Scalars['Boolean']>;
  is_payment_set_up?: InputMaybe<Scalars['Boolean']>;
  lastSyncedAt?: InputMaybe<Scalars['timestamptz']>;
  rules_sheet?: InputMaybe<Scalars['String']>;
  smart_sheet_api_token?: InputMaybe<Scalars['String']>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
  userId?: InputMaybe<Scalars['uuid']>;
};

/** update columns of table "smartsheet" */
export enum Smartsheet_Update_Column {
  /** column name */
  CreatedAt = 'createdAt',
  /** column name */
  Frequency = 'frequency',
  /** column name */
  Id = 'id',
  /** column name */
  IsMonthlyPayment = 'is_monthly_payment',
  /** column name */
  IsPaymentSetUp = 'is_payment_set_up',
  /** column name */
  LastSyncedAt = 'lastSyncedAt',
  /** column name */
  RulesSheet = 'rules_sheet',
  /** column name */
  SmartSheetApiToken = 'smart_sheet_api_token',
  /** column name */
  UpdatedAt = 'updatedAt',
  /** column name */
  UserId = 'userId'
}

export type Smartsheet_Updates = {
  /** sets the columns of the filtered rows to the given values */
  _set?: InputMaybe<Smartsheet_Set_Input>;
  /** filter the rows which have to be updated */
  where: Smartsheet_Bool_Exp;
};

export type Subscription_Root = {
  __typename?: 'subscription_root';
  /** fetch data from the table: "frequency" */
  frequency: Array<Frequency>;
  /** fetch aggregated fields from the table: "frequency" */
  frequency_aggregate: Frequency_Aggregate;
  /** fetch data from the table: "frequency" using primary key columns */
  frequency_by_pk?: Maybe<Frequency>;
  /** fetch data from the table in a streaming manner: "frequency" */
  frequency_stream: Array<Frequency>;
  /** fetch data from the table: "smartsheet" */
  smartsheet: Array<Smartsheet>;
  /** fetch aggregated fields from the table: "smartsheet" */
  smartsheet_aggregate: Smartsheet_Aggregate;
  /** fetch data from the table: "smartsheet" using primary key columns */
  smartsheet_by_pk?: Maybe<Smartsheet>;
  /** fetch data from the table in a streaming manner: "smartsheet" */
  smartsheet_stream: Array<Smartsheet>;
  /** fetch data from the table: "user" */
  user: Array<User>;
  /** fetch aggregated fields from the table: "user" */
  user_aggregate: User_Aggregate;
  /** fetch data from the table: "user" using primary key columns */
  user_by_pk?: Maybe<User>;
  /** fetch data from the table in a streaming manner: "user" */
  user_stream: Array<User>;
};


export type Subscription_RootFrequencyArgs = {
  distinct_on?: InputMaybe<Array<Frequency_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Frequency_Order_By>>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};


export type Subscription_RootFrequency_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Frequency_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Frequency_Order_By>>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};


export type Subscription_RootFrequency_By_PkArgs = {
  title: Scalars['String'];
};


export type Subscription_RootFrequency_StreamArgs = {
  batch_size: Scalars['Int'];
  cursor: Array<InputMaybe<Frequency_Stream_Cursor_Input>>;
  where?: InputMaybe<Frequency_Bool_Exp>;
};


export type Subscription_RootSmartsheetArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


export type Subscription_RootSmartsheet_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


export type Subscription_RootSmartsheet_By_PkArgs = {
  id: Scalars['uuid'];
};


export type Subscription_RootSmartsheet_StreamArgs = {
  batch_size: Scalars['Int'];
  cursor: Array<InputMaybe<Smartsheet_Stream_Cursor_Input>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


export type Subscription_RootUserArgs = {
  distinct_on?: InputMaybe<Array<User_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<User_Order_By>>;
  where?: InputMaybe<User_Bool_Exp>;
};


export type Subscription_RootUser_AggregateArgs = {
  distinct_on?: InputMaybe<Array<User_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<User_Order_By>>;
  where?: InputMaybe<User_Bool_Exp>;
};


export type Subscription_RootUser_By_PkArgs = {
  id: Scalars['uuid'];
};


export type Subscription_RootUser_StreamArgs = {
  batch_size: Scalars['Int'];
  cursor: Array<InputMaybe<User_Stream_Cursor_Input>>;
  where?: InputMaybe<User_Bool_Exp>;
};

/** Boolean expression to compare columns of type "timestamptz". All fields are combined with logical 'AND'. */
export type Timestamptz_Comparison_Exp = {
  _eq?: InputMaybe<Scalars['timestamptz']>;
  _gt?: InputMaybe<Scalars['timestamptz']>;
  _gte?: InputMaybe<Scalars['timestamptz']>;
  _in?: InputMaybe<Array<Scalars['timestamptz']>>;
  _is_null?: InputMaybe<Scalars['Boolean']>;
  _lt?: InputMaybe<Scalars['timestamptz']>;
  _lte?: InputMaybe<Scalars['timestamptz']>;
  _neq?: InputMaybe<Scalars['timestamptz']>;
  _nin?: InputMaybe<Array<Scalars['timestamptz']>>;
};

/** columns and relationships of "user" */
export type User = {
  __typename?: 'user';
  createdAt: Scalars['timestamptz'];
  email: Scalars['String'];
  firstName?: Maybe<Scalars['String']>;
  id: Scalars['uuid'];
  isActive: Scalars['Boolean'];
  lastName: Scalars['String'];
  password?: Maybe<Scalars['String']>;
  profileImage?: Maybe<Scalars['String']>;
  provider?: Maybe<Scalars['String']>;
  smart_sheet_mgt_is_active?: Maybe<Scalars['Boolean']>;
  /** An array relationship */
  smartsheets: Array<Smartsheet>;
  /** An aggregate relationship */
  smartsheets_aggregate: Smartsheet_Aggregate;
  updatedAt: Scalars['timestamptz'];
};


/** columns and relationships of "user" */
export type UserSmartsheetsArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};


/** columns and relationships of "user" */
export type UserSmartsheets_AggregateArgs = {
  distinct_on?: InputMaybe<Array<Smartsheet_Select_Column>>;
  limit?: InputMaybe<Scalars['Int']>;
  offset?: InputMaybe<Scalars['Int']>;
  order_by?: InputMaybe<Array<Smartsheet_Order_By>>;
  where?: InputMaybe<Smartsheet_Bool_Exp>;
};

/** aggregated selection of "user" */
export type User_Aggregate = {
  __typename?: 'user_aggregate';
  aggregate?: Maybe<User_Aggregate_Fields>;
  nodes: Array<User>;
};

/** aggregate fields of "user" */
export type User_Aggregate_Fields = {
  __typename?: 'user_aggregate_fields';
  count: Scalars['Int'];
  max?: Maybe<User_Max_Fields>;
  min?: Maybe<User_Min_Fields>;
};


/** aggregate fields of "user" */
export type User_Aggregate_FieldsCountArgs = {
  columns?: InputMaybe<Array<User_Select_Column>>;
  distinct?: InputMaybe<Scalars['Boolean']>;
};

/** Boolean expression to filter rows from the table "user". All fields are combined with a logical 'AND'. */
export type User_Bool_Exp = {
  _and?: InputMaybe<Array<User_Bool_Exp>>;
  _not?: InputMaybe<User_Bool_Exp>;
  _or?: InputMaybe<Array<User_Bool_Exp>>;
  createdAt?: InputMaybe<Timestamptz_Comparison_Exp>;
  email?: InputMaybe<String_Comparison_Exp>;
  firstName?: InputMaybe<String_Comparison_Exp>;
  id?: InputMaybe<Uuid_Comparison_Exp>;
  isActive?: InputMaybe<Boolean_Comparison_Exp>;
  lastName?: InputMaybe<String_Comparison_Exp>;
  password?: InputMaybe<String_Comparison_Exp>;
  profileImage?: InputMaybe<String_Comparison_Exp>;
  provider?: InputMaybe<String_Comparison_Exp>;
  smart_sheet_mgt_is_active?: InputMaybe<Boolean_Comparison_Exp>;
  smartsheets?: InputMaybe<Smartsheet_Bool_Exp>;
  smartsheets_aggregate?: InputMaybe<Smartsheet_Aggregate_Bool_Exp>;
  updatedAt?: InputMaybe<Timestamptz_Comparison_Exp>;
};

/** unique or primary key constraints on table "user" */
export enum User_Constraint {
  /** unique or primary key constraint on columns "id" */
  UserPkey = 'user_pkey'
}

/** input type for inserting data into table "user" */
export type User_Insert_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  email?: InputMaybe<Scalars['String']>;
  firstName?: InputMaybe<Scalars['String']>;
  id?: InputMaybe<Scalars['uuid']>;
  isActive?: InputMaybe<Scalars['Boolean']>;
  lastName?: InputMaybe<Scalars['String']>;
  password?: InputMaybe<Scalars['String']>;
  profileImage?: InputMaybe<Scalars['String']>;
  provider?: InputMaybe<Scalars['String']>;
  smart_sheet_mgt_is_active?: InputMaybe<Scalars['Boolean']>;
  smartsheets?: InputMaybe<Smartsheet_Arr_Rel_Insert_Input>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
};

/** aggregate max on columns */
export type User_Max_Fields = {
  __typename?: 'user_max_fields';
  createdAt?: Maybe<Scalars['timestamptz']>;
  email?: Maybe<Scalars['String']>;
  firstName?: Maybe<Scalars['String']>;
  id?: Maybe<Scalars['uuid']>;
  lastName?: Maybe<Scalars['String']>;
  password?: Maybe<Scalars['String']>;
  profileImage?: Maybe<Scalars['String']>;
  provider?: Maybe<Scalars['String']>;
  updatedAt?: Maybe<Scalars['timestamptz']>;
};

/** aggregate min on columns */
export type User_Min_Fields = {
  __typename?: 'user_min_fields';
  createdAt?: Maybe<Scalars['timestamptz']>;
  email?: Maybe<Scalars['String']>;
  firstName?: Maybe<Scalars['String']>;
  id?: Maybe<Scalars['uuid']>;
  lastName?: Maybe<Scalars['String']>;
  password?: Maybe<Scalars['String']>;
  profileImage?: Maybe<Scalars['String']>;
  provider?: Maybe<Scalars['String']>;
  updatedAt?: Maybe<Scalars['timestamptz']>;
};

/** response of any mutation on the table "user" */
export type User_Mutation_Response = {
  __typename?: 'user_mutation_response';
  /** number of rows affected by the mutation */
  affected_rows: Scalars['Int'];
  /** data from the rows affected by the mutation */
  returning: Array<User>;
};

/** input type for inserting object relation for remote table "user" */
export type User_Obj_Rel_Insert_Input = {
  data: User_Insert_Input;
  /** upsert condition */
  on_conflict?: InputMaybe<User_On_Conflict>;
};

/** on_conflict condition type for table "user" */
export type User_On_Conflict = {
  constraint: User_Constraint;
  update_columns?: Array<User_Update_Column>;
  where?: InputMaybe<User_Bool_Exp>;
};

/** Ordering options when selecting data from "user". */
export type User_Order_By = {
  createdAt?: InputMaybe<Order_By>;
  email?: InputMaybe<Order_By>;
  firstName?: InputMaybe<Order_By>;
  id?: InputMaybe<Order_By>;
  isActive?: InputMaybe<Order_By>;
  lastName?: InputMaybe<Order_By>;
  password?: InputMaybe<Order_By>;
  profileImage?: InputMaybe<Order_By>;
  provider?: InputMaybe<Order_By>;
  smart_sheet_mgt_is_active?: InputMaybe<Order_By>;
  smartsheets_aggregate?: InputMaybe<Smartsheet_Aggregate_Order_By>;
  updatedAt?: InputMaybe<Order_By>;
};

/** primary key columns input for table: user */
export type User_Pk_Columns_Input = {
  id: Scalars['uuid'];
};

/** select columns of table "user" */
export enum User_Select_Column {
  /** column name */
  CreatedAt = 'createdAt',
  /** column name */
  Email = 'email',
  /** column name */
  FirstName = 'firstName',
  /** column name */
  Id = 'id',
  /** column name */
  IsActive = 'isActive',
  /** column name */
  LastName = 'lastName',
  /** column name */
  Password = 'password',
  /** column name */
  ProfileImage = 'profileImage',
  /** column name */
  Provider = 'provider',
  /** column name */
  SmartSheetMgtIsActive = 'smart_sheet_mgt_is_active',
  /** column name */
  UpdatedAt = 'updatedAt'
}

/** input type for updating data in table "user" */
export type User_Set_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  email?: InputMaybe<Scalars['String']>;
  firstName?: InputMaybe<Scalars['String']>;
  id?: InputMaybe<Scalars['uuid']>;
  isActive?: InputMaybe<Scalars['Boolean']>;
  lastName?: InputMaybe<Scalars['String']>;
  password?: InputMaybe<Scalars['String']>;
  profileImage?: InputMaybe<Scalars['String']>;
  provider?: InputMaybe<Scalars['String']>;
  smart_sheet_mgt_is_active?: InputMaybe<Scalars['Boolean']>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
};

/** Streaming cursor of the table "user" */
export type User_Stream_Cursor_Input = {
  /** Stream column input with initial value */
  initial_value: User_Stream_Cursor_Value_Input;
  /** cursor ordering */
  ordering?: InputMaybe<Cursor_Ordering>;
};

/** Initial value of the column from where the streaming should start */
export type User_Stream_Cursor_Value_Input = {
  createdAt?: InputMaybe<Scalars['timestamptz']>;
  email?: InputMaybe<Scalars['String']>;
  firstName?: InputMaybe<Scalars['String']>;
  id?: InputMaybe<Scalars['uuid']>;
  isActive?: InputMaybe<Scalars['Boolean']>;
  lastName?: InputMaybe<Scalars['String']>;
  password?: InputMaybe<Scalars['String']>;
  profileImage?: InputMaybe<Scalars['String']>;
  provider?: InputMaybe<Scalars['String']>;
  smart_sheet_mgt_is_active?: InputMaybe<Scalars['Boolean']>;
  updatedAt?: InputMaybe<Scalars['timestamptz']>;
};

/** update columns of table "user" */
export enum User_Update_Column {
  /** column name */
  CreatedAt = 'createdAt',
  /** column name */
  Email = 'email',
  /** column name */
  FirstName = 'firstName',
  /** column name */
  Id = 'id',
  /** column name */
  IsActive = 'isActive',
  /** column name */
  LastName = 'lastName',
  /** column name */
  Password = 'password',
  /** column name */
  ProfileImage = 'profileImage',
  /** column name */
  Provider = 'provider',
  /** column name */
  SmartSheetMgtIsActive = 'smart_sheet_mgt_is_active',
  /** column name */
  UpdatedAt = 'updatedAt'
}

export type User_Updates = {
  /** sets the columns of the filtered rows to the given values */
  _set?: InputMaybe<User_Set_Input>;
  /** filter the rows which have to be updated */
  where: User_Bool_Exp;
};

/** Boolean expression to compare columns of type "uuid". All fields are combined with logical 'AND'. */
export type Uuid_Comparison_Exp = {
  _eq?: InputMaybe<Scalars['uuid']>;
  _gt?: InputMaybe<Scalars['uuid']>;
  _gte?: InputMaybe<Scalars['uuid']>;
  _in?: InputMaybe<Array<Scalars['uuid']>>;
  _is_null?: InputMaybe<Scalars['Boolean']>;
  _lt?: InputMaybe<Scalars['uuid']>;
  _lte?: InputMaybe<Scalars['uuid']>;
  _neq?: InputMaybe<Scalars['uuid']>;
  _nin?: InputMaybe<Array<Scalars['uuid']>>;
};

export type InsertOrUpdateUsersMutationVariables = Exact<{
  users: Array<User_Insert_Input> | User_Insert_Input;
}>;


export type InsertOrUpdateUsersMutation = { __typename?: 'mutation_root', insert_user?: { __typename?: 'user_mutation_response', returning: Array<{ __typename?: 'user', id: any, firstName?: string | null, email: string, isActive: boolean, lastName: string, password?: string | null, profileImage?: string | null, provider?: string | null, updatedAt: any, createdAt: any }> } | null };

export type CheckIfUserExistQueryVariables = Exact<{
  email?: InputMaybe<Scalars['String']>;
  isActive?: InputMaybe<Scalars['Boolean']>;
}>;


export type CheckIfUserExistQuery = { __typename?: 'query_root', user: Array<{ __typename?: 'user', id: any, firstName?: string | null, email: string, isActive: boolean, lastName: string, password?: string | null, profileImage?: string | null, provider?: string | null, updatedAt: any, createdAt: any }> };

export type UserFragment = { __typename?: 'user', id: any, firstName?: string | null, email: string, isActive: boolean, lastName: string, password?: string | null, profileImage?: string | null, provider?: string | null, updatedAt: any, createdAt: any };

export const UserFragmentDoc = gql`
    fragment user on user {
  id
  firstName
  email
  isActive
  lastName
  password
  profileImage
  provider
  updatedAt
  createdAt
}
    `;
export const InsertOrUpdateUsersDocument = gql`
    mutation InsertOrUpdateUsers($users: [user_insert_input!]!) {
  insert_user(
    objects: $users
    on_conflict: {constraint: user_pkey, update_columns: [email, firstName, lastName, password, provider, profileImage, isActive, updatedAt]}
  ) {
    returning {
      ...user
    }
  }
}
    ${UserFragmentDoc}`;
export type InsertOrUpdateUsersMutationFn = Apollo.MutationFunction<InsertOrUpdateUsersMutation, InsertOrUpdateUsersMutationVariables>;

/**
 * __useInsertOrUpdateUsersMutation__
 *
 * To run a mutation, you first call `useInsertOrUpdateUsersMutation` within a React component and pass it any options that fit your needs.
 * When your component renders, `useInsertOrUpdateUsersMutation` returns a tuple that includes:
 * - A mutate function that you can call at any time to execute the mutation
 * - An object with fields that represent the current status of the mutation's execution
 *
 * @param baseOptions options that will be passed into the mutation, supported options are listed on: https://www.apollographql.com/docs/react/api/react-hooks/#options-2;
 *
 * @example
 * const [insertOrUpdateUsersMutation, { data, loading, error }] = useInsertOrUpdateUsersMutation({
 *   variables: {
 *      users: // value for 'users'
 *   },
 * });
 */
export function useInsertOrUpdateUsersMutation(baseOptions?: Apollo.MutationHookOptions<InsertOrUpdateUsersMutation, InsertOrUpdateUsersMutationVariables>) {
        const options = {...defaultOptions, ...baseOptions}
        return Apollo.useMutation<InsertOrUpdateUsersMutation, InsertOrUpdateUsersMutationVariables>(InsertOrUpdateUsersDocument, options);
      }
export type InsertOrUpdateUsersMutationHookResult = ReturnType<typeof useInsertOrUpdateUsersMutation>;
export type InsertOrUpdateUsersMutationResult = Apollo.MutationResult<InsertOrUpdateUsersMutation>;
export type InsertOrUpdateUsersMutationOptions = Apollo.BaseMutationOptions<InsertOrUpdateUsersMutation, InsertOrUpdateUsersMutationVariables>;
export const CheckIfUserExistDocument = gql`
    query checkIfUserExist($email: String, $isActive: Boolean) {
  user(where: {email: {_eq: $email}, isActive: {_eq: $isActive}}) {
    ...user
  }
}
    ${UserFragmentDoc}`;

/**
 * __useCheckIfUserExistQuery__
 *
 * To run a query within a React component, call `useCheckIfUserExistQuery` and pass it any options that fit your needs.
 * When your component renders, `useCheckIfUserExistQuery` returns an object from Apollo Client that contains loading, error, and data properties
 * you can use to render your UI.
 *
 * @param baseOptions options that will be passed into the query, supported options are listed on: https://www.apollographql.com/docs/react/api/react-hooks/#options;
 *
 * @example
 * const { data, loading, error } = useCheckIfUserExistQuery({
 *   variables: {
 *      email: // value for 'email'
 *      isActive: // value for 'isActive'
 *   },
 * });
 */
export function useCheckIfUserExistQuery(baseOptions?: Apollo.QueryHookOptions<CheckIfUserExistQuery, CheckIfUserExistQueryVariables>) {
        const options = {...defaultOptions, ...baseOptions}
        return Apollo.useQuery<CheckIfUserExistQuery, CheckIfUserExistQueryVariables>(CheckIfUserExistDocument, options);
      }
export function useCheckIfUserExistLazyQuery(baseOptions?: Apollo.LazyQueryHookOptions<CheckIfUserExistQuery, CheckIfUserExistQueryVariables>) {
          const options = {...defaultOptions, ...baseOptions}
          return Apollo.useLazyQuery<CheckIfUserExistQuery, CheckIfUserExistQueryVariables>(CheckIfUserExistDocument, options);
        }
export type CheckIfUserExistQueryHookResult = ReturnType<typeof useCheckIfUserExistQuery>;
export type CheckIfUserExistLazyQueryHookResult = ReturnType<typeof useCheckIfUserExistLazyQuery>;
export type CheckIfUserExistQueryResult = Apollo.QueryResult<CheckIfUserExistQuery, CheckIfUserExistQueryVariables>;