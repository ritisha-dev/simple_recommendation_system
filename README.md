#### Simple Book Recommendation System

The current project aims to build a simple book recommendation system using current user-item preferences. We use cosine similarity based item matrix to find the most relevant recommendations for a user_id input. 

##### Approach

###### Data

We have 3 distinct datasets provided:
    1. User: Consists of unique users information, 278k users
    2. Books: Consists of unique books information, 271k books
    3, Ratings: Consists User-Book rating information, 1.1M ratings info (high null ratings)

###### Preprocessing

The data consists of duplicates, spaces, etc. and needs to be cleansed to enhance the data quality and be able to use it. The Preprocess module in preprocess.py aims to achieve the following:
    1. Convert all column names to lower case
    2. Fix data types
    3. Remove whitespaces
    4. Remove duplicates

###### Feature Engineering

After the data is cleansed we are now ready to create useful features using the Feat module in feat.py. We create user level features and book level features like:
    User
        1. # of ratings given by user
        2. mean rating given by user

    Book
        1. # of ratings for the book
        2. mean of ratings for the book
        3. weighted # of ratings for the book

    Cosine Similarity
        1. Cosine similarity between books based on user ratings

Although, the User and Book features are not being currently used in the recommendations, we should find use for them when implementing a LTR system. 

##### Recommendations

We can now recommend books to users based on 2 criteria:
    1. Cosine similarity based lowest distance to previously read books
    2. Most popular books for users with no historical data - Popularity based

##### Execution

We can request recommendations from the model using the `run.py` script from commandline. It requires 3 commandline parameters:

    1. data path specified with --path
    2. training flag with --training
    3. user_id with --user_id

Ex: `run.py --path data --training False --user_id 2`

The output looks as follows:

```Scoring data
  user_id        isbn  rank                                       book_title
0       2  0971880107     1                                      Wild Animus
1       2  0316666343     2                        The Lovely Bones: A Novel
2       2  0385504209     3                                The Da Vinci Code
3       2  0060928336     4  Divine Secrets of the Ya-Ya Sisterhood: A Novel
4       2  0312195516     5              The Red Tent (Bestselling Backlist)
```
