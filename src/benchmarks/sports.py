import pandas as pd

# Teams Table
teams_data = {
    "team_id": [1, 2, 3, 4, 5, 6],
    "team_name": ["Lakers", "Celtics", "Bulls", "Warriors", "Heat", "Nets"],
    "city": ["Los Angeles", "Boston", "Chicago", "San Francisco", "Miami", "Brooklyn"],
    "annual_budget": [150000000, 142000000, 128000000, 165000000, 135000000, 140000000],
    "championships": [17, 17, 6, 7, 3, 0],
}
aljksdfhljkahsdjlfhal = pd.DataFrame(teams_data)

# Players Table
players_data = {
    "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": [
        "LeBron James",
        "Stephen Curry",
        "Kevin Durant",
        "Jayson Tatum",
        "Jimmy Butler",
        "DeMar DeRozan",
        "Klay Thompson",
        "Bam Adebayo",
    ],
    "team_id": [1, 4, 6, 2, 5, 3, 4, 5],
    "position": ["SF", "PG", "SF", "SF", "SF", "SG", "SG", "C"],
    "salary": [
        44500000,
        48070014,
        44119845,
        37084800,
        37653300,
        28000000,
        40600000,
        32600000,
    ],
    "age": [38, 35, 34, 25, 33, 33, 33, 26],
}
asdhjfhjksdhakjhask = pd.DataFrame(players_data)

# Coaches Table
coaches_data = {
    "coach_id": [1, 2, 3, 4, 5, 6],
    "name": [
        "Darvin Ham",
        "Joe Mazzulla",
        "Billy Donovan",
        "Steve Kerr",
        "Erik Spoelstra",
        "Jacque Vaughn",
    ],
    "team_id": [1, 2, 3, 4, 5, 6],
    "salary": [4000000, 3500000, 5000000, 8500000, 8500000, 4000000],
    "years_experience": [2, 2, 8, 9, 15, 4],
}
yuieywuijhskjdf = pd.DataFrame(coaches_data)

# Team Statistics Table
team_stats_data = {
    "team_id": [1, 2, 3, 4, 5, 6],
    "wins": [43, 57, 40, 44, 44, 45],
    "losses": [39, 25, 42, 38, 38, 37],
    "points_per_game": [117.2, 117.9, 113.1, 118.9, 109.5, 113.4],
    "rebounds_per_game": [45.7, 47.5, 42.9, 44.6, 41.9, 40.5],
    "assists_per_game": [25.3, 26.7, 24.5, 29.8, 23.8, 25.5],
}
dajklshdjklahsjkldh = pd.DataFrame(team_stats_data)

# Player Statistics Table
player_stats_data = {
    "player_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "points_per_game": [28.9, 29.4, 29.1, 30.1, 22.9, 24.5, 21.9, 20.4],
    "rebounds_per_game": [8.3, 6.1, 6.7, 8.8, 5.9, 4.6, 4.1, 9.2],
    "assists_per_game": [6.8, 6.3, 5.0, 4.6, 5.3, 5.1, 2.4, 3.2],
    "minutes_per_game": [35.5, 34.7, 36.0, 37.6, 33.8, 34.0, 33.6, 32.9],
    "field_goal_percentage": [0.500, 0.485, 0.560, 0.466, 0.538, 0.504, 0.436, 0.540],
}
kljkljaklsjdkjasldk = pd.DataFrame(player_stats_data)


def get_metadata():
    return """This database includes:

1. Teams Table:
   - Basic team information
   - Annual budget
   - Number of championships

2. Players Table:
   - Player information
   - Current team (linked via team_id)
   - Salary
   - Position and age

3. Coaches Table:
   - Coach information
   - Current team (linked via team_id)
   - Salary
   - Years of experience

4. Team Statistics Table:
   - Team performance metrics
   - Win/loss records
   - Team averages

5. Player Statistics Table:
   - Individual player performance metrics
   - Scoring, rebounding, and assist averages
   - Playing time and shooting efficiency

The tables are related through:
- team_id (linking teams, players, coaches, and team statistics)
- player_id (linking players and player statistics)"""


dbs = {
    "teams.tbl": aljksdfhljkahsdjlfhal,
    "coach.tbl": yuieywuijhskjdf,
    "player.tbl": asdhjfhjksdhakjhask,
    "team_stats.tbl": dajklshdjklahsjkldh,
    "player_states.tbl": kljkljaklsjdkjasldk,
}


def read_db(name):
    return dbs[name]


def ls():
    s = ""
    for k in dbs.keys():
        s += k + "\n"
    return s


def select(df, columns=None, conditions=None):
    """
    Select specific columns from a table with optional conditions

    Args:
        df (DataFrame): Input DataFrame
        columns (list): List of columns to select. If None, selects all columns
        conditions (dict): Dictionary of column-value pairs for filtering. The value is either an exact match or a tuple containing an operator (e.g. '<') and a value.

    Returns:
        DataFrame: Filtered DataFrame
    """

    if columns:
        df = df[columns]

    if conditions:
        mask = True
        for col, value in conditions.items():
            if isinstance(value, tuple):
                op, val = value
                if op == ">":
                    mask = mask & (df[col] > val)
                elif op == "<":
                    mask = mask & (df[col] < val)
                elif op == ">=":
                    mask = mask & (df[col] >= val)
                elif op == "<=":
                    mask = mask & (df[col] <= val)
                elif op == "!=":
                    mask = mask & (df[col] != val)
            else:
                mask = mask & (df[col] == value)
        df = df[mask]

    return df


def sort(df, columns, ascending=True):
    """
    Sort DataFrame by specified columns

    Args:
        df (DataFrame): Input DataFrame
        columns (str or list): Column(s) to sort by
        ascending (bool or list): Sort order

    Returns:
        DataFrame: Sorted DataFrame
    """
    return df.sort_values(columns, ascending=ascending)


def merge_tables(df1, df2, on, how="inner"):
    """
    Merge two tables

    Args:
        df1 (DataFrame): Input DataFrame
        df2 (DataFrame): Input DataFrame
        on (str or list): Column(s) to join on
        how (str): Type of merge (inner, outer, left, right)

    Returns:
        DataFrame: Merged DataFrame
    """

    return pd.merge(df1, df2, on=on, how=how)


def aggregate(df, group_by, agg_dict):
    """
    Perform aggregation operations

    Args:
        df (DataFrame): Input DataFrame
        group_by (str or list): Column(s) to group by
        agg_dict (dict): Dictionary of column-operation pairs

    Returns:
        DataFrame: Aggregated DataFrame
    """
    return df.groupby(group_by).agg(agg_dict)


def limit(df, n):
    """
    Limit the number of rows returned

    Args:
        df (DataFrame): Input DataFrame
        n (int): Number of rows to return

    Returns:
        DataFrame: Limited DataFrame
    """
    return df.head(n)


def calculate_column(df, new_column, expression):
    """
    Add a calculated column to the DataFrame

    Args:
        df (DataFrame): Input DataFrame
        new_column (str): Name of the new column
        expression (str): Python expression to calculate the column

    Returns:
        DataFrame: DataFrame with new column
    """
    df = df.copy()
    try:
        # Safely evaluate the expression using DataFrame.eval
        df[new_column] = df.eval(expression)
    except Exception as e:
        raise ValueError(f"Error in calculating column '{new_column}': {e}")
    return df


function_mapping = {
    "read_db": read_db,
    "select": select,
    "sort": sort,
    "merge_tables": merge_tables,
    "aggregate": aggregate,
    "limit": limit,
    "calculate_column": calculate_column,
    "ls": ls,
    "get_metadata": get_metadata,
}


def execute_operations(operations):
    """
    Execute a list of operations on the database and return the final result.

    Args:
        operations (list): List of operation dictionaries. Each dictionary should have:
            - 'function': The name of the function to execute (str)
            - 'args': A dictionary of arguments for the function

    Returns:
        DataFrame or str: The result after executing all operations. Could be a DataFrame or string (e.g., from 'ls' or 'get_metadata')
    """
    current_df = None
    for idx, op in enumerate(operations):
        func_name = op.get("function")
        args = op.get("args", {})

        if func_name not in function_mapping:
            raise ValueError(f"Function '{func_name}' is not supported.")

        func = function_mapping[func_name]

        if func_name == "read_db":
            current_df = func(**args)

        elif func_name in ["ls", "get_metadata"]:
            # These functions do not require a DataFrame
            return func()

        elif func_name == "merge_tables":
            # Expecting a 'table' argument to merge with
            if "df2" in args:
                current_df = func(current_df, args.pop("df2"), **args)
            else:
                table_name = args.pop("table")
                additional_df = read_db(table_name)
                current_df = func(current_df, additional_df, **args)

        else:
            if current_df is None:
                raise ValueError(
                    f"No DataFrame is currently loaded. Ensure the first operation is 'read_db'."
                )
            current_df = func(current_df, **args)

    return current_df


QUESTIONS = {
    "Show me all the teams that are based in Los Angeles. Keep all columns.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "teams.tbl"}},
            {"function": "select", "args": {"conditions": {"city": "Los Angeles"}}},
        ]
    ),
    "All players who earn more than $40,000,000 annually. Keep all columns.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {"function": "select", "args": {"conditions": {"salary": (">", 40000000)}}},
        ]
    ),
    "Show the Top 3 Highest-Paid Coaches, keep the name and salary columns only. Rows should be sorted by pay (highest to lowest)": execute_operations(
        [
            {"function": "read_db", "args": {"name": "coach.tbl"}},
            {"function": "sort", "args": {"columns": "salary", "ascending": False}},
            {"function": "limit", "args": {"n": 3}},
            {"function": "select", "args": {"columns": ["name", "salary"]}},
        ]
    ),
    "Get me a table with a player's name and their team name. Only these columns. Sort in alphabetical order by first name.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {
                "function": "merge_tables",
                "args": {"on": "team_id", "how": "inner", "table": "teams.tbl"},
            },
            {"function": "select", "args": {"columns": ["name", "team_name"]}},
            {"function": "sort", "args": {"columns": "name", "ascending": True}},
        ]
    ),
    "What is the average salary of players in each team? I only want the table to have team name and salary columns. Sort the rows from highest to lowest avg salary.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {
                "function": "aggregate",
                "args": {"group_by": "team_id", "agg_dict": {"salary": "mean"}},
            },
            {
                "function": "merge_tables",
                "args": {"on": "team_id", "how": "left", "table": "teams.tbl"},
            },
            {"function": "select", "args": {"columns": ["team_name", "salary"]}},
            {"function": "sort", "args": {"columns": "salary", "ascending": False}}
        ]
    ),
    "Show all players who are over 30 years old and play as Shooting Guards. Leave only name, age and position columns.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {
                "function": "select",
                "args": {"conditions": {"age": (">", 30), "position": "SG"}},
            },
            {"function": "select", "args": {"columns": ["name", "age", "position"]}},
        ]
    ),
    "Show Top 5 Teams with the Most Wins, should be sorted from most wins to least wins. Only show team name and wins.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "team_stats.tbl"}},
            {
                "function": "merge_tables",
                "args": {"on": "team_id", "how": "inner", "table": "teams.tbl"},
            },
            {"function": "sort", "args": {"columns": "wins", "ascending": False}},
            {"function": "limit", "args": {"n": 5}},
            {"function": "select", "args": {"columns": ["team_name", "wins"]}},
        ]
    ),
    "Calculate a New Column for Player Salary in Millions. I want the table to have 3 columns: name, salary, and salary_millions. Sort from highest to lowest salary.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {
                "function": "calculate_column",
                "args": {"new_column": "salary_millions", "expression": "salary / 1e6"},
            },
            {
                "function": "select",
                "args": {"columns": ["name", "salary", "salary_millions"]},
            },
            {"function": "sort", "args": {"columns": "salary", "ascending": False}}
        ]
    ),
    "Show all coaches who have more than ten years of coaching experience. Leave just the 2 columns (name and year experience). Sort by most experience to least experience.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "coach.tbl"}},
            {
                "function": "select",
                "args": {"conditions": {"years_experience": (">", 10)}},
            },
            {"function": "select", "args": {"columns": ["name", "years_experience"]}},
            {"function": "sort", "args": {"columns": "years_experience", "ascending": False}}
        ]
    ),
    "Provide the metadata information for the entire database.": execute_operations(
        [{"function": "get_metadata", "args": {}}]
    ),
    "Show each player alongside the annual budget of their team. Table should have the players name, team, and annual budget. Sort from highest annual budget to lowest.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "player.tbl"}},
            {
                "function": "merge_tables",
                "args": {"on": "team_id", "how": "inner", "table": "teams.tbl"},
            },
            {
                "function": "select",
                "args": {"columns": ["name", "team_name", "annual_budget"]},
            },
            {"function": "sort", "args": {"columns": "annual_budget", "ascending": False}}
        ]
    ),
    "Identify teams that have coaches with over fifteen years of experience. Show name, team, and years of experience.": execute_operations(
        [
            {"function": "read_db", "args": {"name": "coach.tbl"}},
            {
                "function": "select",
                "args": {"conditions": {"years_experience": (">", 15)}},
            },
            {
                "function": "merge_tables",
                "args": {"on": "team_id", "how": "inner", "table": "teams.tbl"},
            },
            {
                "function": "select",
                "args": {"columns": ["name", "team_name", "years_experience"]},
            },
        ]
    ),
}
