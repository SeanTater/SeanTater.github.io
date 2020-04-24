---
layout: post
title:  "Finding data types in SQLite, Teradata, Hive, and Phoenix"
date:   2020-04-22 10:23:44 -0400
categories: database
---

Quick reference for finding types for your DESCRIBE.

## Hive 

For Hive, you can get a list of column names and types using:
```sql
    -- List databases
    SHOW DATABASES;
    -- List tables for each database
    SHOW TABLES IN database_name;
    -- List columns for each table
    DESCRIBE your_table_name;
```
and the results for the final one (which lists only one table) will look like the following:

<table-wrap markdown="block">

| col_name     | data_type | comment                |
| ------------ | --------- | ---------------------- |
|entry_id      |int        | Unique ID              |
|name          |string     |                        |
|product_id    |string     | Foreign key to Product |
|item_id       |int        | Foreign key to Item    |
|vendor_id     |int        | Not sure..             |

</table-wrap>

There's also `DESCRIBE FORMATTED`, which gives you a lot of information about each table, but in a
challenging to parse format intended for human consumption.

## SQLite

For SQLite, there is always `.schema` in the CLI, which will get you the text for your CREATE TABLE
but that's not really suitable for use in scripts.

```sql
-- sqlite> .schema sqlite_master
CREATE TABLE sqlite_master (
  type text,
  name text,
  tbl_name text,
  rootpage integer,
  sql text
);
```

In contrast, you can easily get most of what you normally need from a `pragma`:
```sql
-- Anywhere you can submit SQL to SQLite
PRAGMA table_info(sqlite_master);
```

<table-wrap markdown="block">

cid        | name      | type     | notnull   | dflt_value | pk        
---------- |-----------|----------|---------- |----------  |----------
0          |type       |text      |0          |            | 0         
1          |name       |text      |0          |            | 0         
2          |tbl_name   |text      |0          |            | 0         
3          |rootpage   |int       |0          |            | 0         
4          |sql        |text      |0          |            | 0         

</table-wrap>

## Phoenix
Similar to SQLite, there are multiple ways to get schemata. If you use `sqlline` or something of a
similar flavor, there's `!tables`, and for one table there is `!describe table_name`.
In some cases though (like mine), you will need to be able to access this programmatically. So
instead you should opt to read from `system.catalog`

```sql
SELECT * FROM system.catalog;
```

The trouble is, it will give you so much more information than you were ever ready for.
I can't even show the table here because it is too wide to format reasonably. It contains:

```
TENANT_ID, TABLE_SCHEM, TABLE_NAME, COLUMN_NAME, COLUMN_FAMILY, TABLE_SEQ_NUM, TABLE_TYPE, PK_NAME, COLUMN_COUNT, SALT_BUCKETS, DATA_TABLE_NAME, INDEX_STATE, IMMUTABLE_ROWS, VIEW_STATEMENT, DEFAULT_COLUMN_FAMILY, DISABLE_WAL, MULTI_TENANT, VIEW_TYPE, VIEW_INDEX_ID, DATA_TYPE, COLUMN_SIZE, DECIMAL_DIGITS, NULLABLE, ORDINAL_POSITION, SORT_ORDER, ARRAY_SIZE, VIEW_CONSTANT, IS_VIEW_REFERENCED, KEY_SEQ, LINK_TYPE, TYPE_NAME, REMARKS, SELF_REFERENCING_COL_NAME, REF_GENERATION, BUFFER_LENGTH, NUM_PREC_RADIX, COLUMN_DEF, SQL_DATA_TYPE, SQL_DATETIME_SUB, CHAR_OCTET_LENGTH, IS_NULLABLE, SCOPE_CATALOG, SCOPE_SCHEMA, SCOPE_TABLE, SOURCE_DATA_TYPE, IS_AUTOINCREMENT, INDEX_TYPE, INDEX_DISABLE_TIMESTAMP, STORE_NULLS, BASE_COLUMN_COUNT, IS_ROW_TIMESTAMP, TRANSACTIONAL, UPDATE_CACHE_FREQUENCY, IS_NAMESPACE_MAPPED
```

## Teradata
Of these four databases, Teradata has the most rigorous type system and cataloging setup.
However, on that account even the simple things, like determining a column's type is very tedious.
These are the queries you would use to find that catalog information, and the columns of the
metadata they return:

> These types can be hard to interpret. Keep scrolling for more information.

```sql
SELECT * FROM DBC.Databases;
-- Will get you:
-- DatabaseName, CreatorName, OwnerName, AccountName, ProtectionType, JournalFlag, PermSpace, SpoolSpace, TempSpace, CommentString, CreateTimeStamp, LastAlterName, LastAlterTimeStamp, DBKind, AccessCount, LastAccessTimeStamp
```

```sql
-- Get one row about each table
SELECT * FROM DBC.Tables;
-- Will get you:
-- DatabaseName, TableName, Version, TableKind, ProtectionType, JournalFlag, CreatorName, RequestText, CommentString, ParentCount, ChildCount, NamedTblCheckCount, UnnamedTblCheckExist, PrimaryKeyIndexId, RepStatus, CreateTimeStamp, LastAlterName, LastAlterTimeStamp, RequestTxtOverflow, AccessCount, LastAccessTimeStamp, UtilVersion, QueueFlag, CommitOpt, TransLog, CheckOpt, TemporalProperty, ResolvedCurrent_Date, ResolvedCurrent_Timestamp, SystemDefinedJI, VTQualifier, TTQualifier
```

```sql
-- Get one row about each table
SELECT * FROM DBC.Columns;
-- Will get you:
-- DatabaseName, TableName, ColumnName, ColumnFormat, ColumnTitle, SPParameterType, ColumnType, ColumnUDTName, ColumnLength, DefaultValue, Nullable, CommentString, DecimalTotalDigits, DecimalFractionalDigits, ColumnId, UpperCaseFlag, Compressible, CompressValue, ColumnConstraint, ConstraintCount, CreatorName, CreateTimeStamp, LastAlterName, LastAlterTimeStamp, CharType, IdColType, AccessCount, LastAccessTimeStamp, CompressValueList, TimeDimension, VTCheckType, TTCheckType, ConstraintId, ArrayColNumberOfDimensions, ArrayColScope, ArrayColElementType, ArrayColElementUdtName, TSColumnType
```

I'll save you the Google search. A [popular forum thread][1] that leads Google search results for
"Teradata column data types" explains the meanings of the `ColumnType` column. You can find further
details on these types in [Teradata's official documentation][2], and you can interpret the codes
to one of those names using this lookup:

<table-wrap markdown="block">

Code  | Long Format
------|------------
A1    | ARRAY 
AN    | MULTI-DIMENSIONAL ARRAY
AT    | TIME 
BF    | BYTE
BO    | BLOB 
BV    | VARBYTE
CF    | CHARACTER 
CO    | CLOB
CV    | VARCHAR 
D     | DECIMAL
DA    | DATE 
DH    | INTERVAL DAY TO HOUR
DM    | INTERVAL DAY TO MINUTE 
DS    | INTERVAL DAY TO SECOND
DY    | INTERVAL DAY 
F     | FLOAT
HM    | INTERVAL HOUR TO MINUTE 
HS    | INTERVAL HOUR TO SECOND
HR    | INTERVAL HOUR 
I     | INTEGER
I1    | BYTEINT 
I2    | SMALLINT
I8    | BIGINT 
JN    | JSON
MI    | INTERVAL MINUTE 
MO    | INTERVAL MONTH
MS    | INTERVAL MINUTE TO SECOND 
N     | NUMBER
PD    | PERIOD(DATE) 
PM    | PERIOD(TIMESTAMP WITH TIME ZONE)
PS    | PERIOD(TIMESTAMP) 
PT    | PERIOD(TIME)
PZ    | PERIOD(TIME WITH TIME ZONE) 
SC    | INTERVAL SECOND
SZ    | TIMESTAMP WITH TIME ZONE 
TS    | TIMESTAMP
TZ    | TIME WITH TIME ZONE 
UT    | UDT Type
XM    | XML 
YM    | INTERVAL YEAR TO MONTH
YR    | INTERVAL YEAR 
++    | TD_ANYTYPE

</table-wrap>

But even then, you can't get column types for views, and where I work, views are very popular.
Hopefully you don't need it too often, but in a pinch, you can still always `SELECT`.
Conveniently, it provides you one of the long forms rather than type codes, and they tend to be the
most general of their type (i.e. `INTEGER` rather than `BIGINT`) but don't ask me why!

```sql
SELECT TOP 1 type(column_name) FROM table_name;
```

Hope this helps!

[1]: https://downloads.teradata.com/forum/database/list-of-all-teradata-column-types-with-their-associated-dbtypes
[2]: https://docs.teradata.com/reader/iRq_F~XxKYWu7Kv~HRd~ew/D_RBrANpKte9E5uvWjq8~Q