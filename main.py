

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, 
    IntegerType, TimestampType, 
    DoubleType, StringType
)
import requests
import os
from datetime import datetime

# Define the yellow taxi schema
yellow_taxi_schema = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("PULocationID", IntegerType(), True),
    StructField("DOLocationID", IntegerType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("congestion_surcharge", DoubleType(), True)
])

def download_taxi_data(taxi_type ,spark, year, month, output_path):
    """
    Download monthly taxi trip data from NYC website
    """
    # Format URL for the specified year and month
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month:02d}.parquet"
    
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_path, f"yellow_tripdata_{year}-{month:02d}.parquet")
    
    try:
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded to {output_file}")
            
            # Read the downloaded parquet file
            df = spark.read.schema(yellow_taxi_schema).parquet(output_file)
            return df
        else:
            print(f"Failed to download data: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None

def download_zone_lookup(spark, output_path):
    """
    Download taxi zone lookup table
    """
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    output_file = os.path.join(output_path, "taxi_zone_lookup.csv")
    
    try:
        print(f"Downloading zone lookup from {url}...")
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded to {output_file}")
            
            # Define schema for zone lookup
            zone_schema = StructType([
                StructField("LocationID", IntegerType(), True),
                StructField("Borough", StringType(), True),
                StructField("Zone", StringType(), True),
                StructField("service_zone", StringType(), True)
            ])
            
            # Read the downloaded CSV
            df = spark.read.schema(zone_schema).csv(output_file, header=True)
            return df
        else:
            print(f"Failed to download zone lookup: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading zone lookup: {str(e)}")
        return None

def process_taxi_data(df):
    """
    Process taxi data to add calculated columns
    """
    return df.withColumn(
        "trip_duration_minutes",
        F.round((F.unix_timestamp("tpep_dropoff_datetime") - 
                F.unix_timestamp("tpep_pickup_datetime")) / 60, 2)
    ).withColumn(
        "speed_mph",
        F.when(
            F.col("trip_duration_minutes") > 0,
            F.round(F.col("trip_distance") / (F.col("trip_duration_minutes") / 60), 2)
        ).otherwise(None)
    ).withColumn(
        "is_weekend",
        F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), True).otherwise(False)
    )

def identify_outliers(df):
    """
    Identify outliers in fare amounts using percentile calculation
    """
    window = Window.partitionBy(F.to_date("tpep_pickup_datetime"))
    
    return df.withColumn(
        "fare_percentile_75",
        F.expr("percentile_approx(fare_amount, 0.75)").over(window)
    ).withColumn(
        "is_outlier",
        F.col("fare_amount") > (F.col("fare_percentile_75") * 1.5)
    )

def create_hourly_aggregations(spark, df):
    """
    Create hourly aggregations
    """
    df.createOrReplaceTempView("taxi_trips")
    
    return spark.sql("""
        SELECT 
            date_trunc('hour', tpep_pickup_datetime) as hour,
            PULocationID,
            COUNT(*) as total_trips,
            SUM(total_amount) as total_revenue,
            AVG(trip_duration_minutes) as avg_duration,
            AVG(speed_mph) as avg_speed
        FROM taxi_trips
        GROUP BY 1, 2
    """)

def create_daily_aggregations(spark, df):
    """
    Create daily aggregations
    """
    df.createOrReplaceTempView("taxi_trips")
    
    return spark.sql("""
        SELECT 
            date(tpep_pickup_datetime) as date,
            PULocationID,
            COUNT(*) as total_trips,
            SUM(total_amount) as total_revenue,
            SUM(CASE WHEN is_outlier THEN 1 ELSE 0 END) as outlier_trips
        FROM taxi_trips
        GROUP BY 1, 2
    """)

def get_top_pickup_locations(spark, zone_df):
    """
    Get top 10 pickup locations by revenue
    """
    zone_df.createOrReplaceTempView("taxi_zone_lookup")
    
    return spark.sql("""
        SELECT 
            t.PULocationID,
            z.Zone as pickup_zone,
            z.Borough as borough,
            SUM(t.total_amount) as total_revenue,
            COUNT(*) as total_trips
        FROM taxi_trips t
        JOIN taxi_zone_lookup z ON t.PULocationID = z.LocationID
        GROUP BY 1, 2, 3
        ORDER BY 4 DESC
        LIMIT 10
    """)

def get_weekday_weekend_patterns(spark):
    """
    Analyze weekday vs weekend trip patterns by month
    """
    return spark.sql("""
        SELECT 
            MONTH(tpep_pickup_datetime) as month,
            is_weekend,
            COUNT(*) as total_trips,
            SUM(total_amount) as total_revenue,
            AVG(trip_duration_minutes) as avg_duration
        FROM taxi_trips
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)

def get_long_trips(spark, zone_df):
    """
    Analyze long trips (> 10 miles)
    """
    zone_df.createOrReplaceTempView("taxi_zone_lookup")
    
    return spark.sql("""
        SELECT 
            t.PULocationID,
            pickup.Zone as pickup_zone,
            pickup.Borough as pickup_borough,
            t.DOLocationID,
            dropoff.Zone as dropoff_zone,
            dropoff.Borough as dropoff_borough,
            COUNT(*) as trip_count,
            AVG(trip_distance) as avg_distance,
            AVG(total_amount) as avg_fare,
            AVG(trip_duration_minutes) as avg_duration
        FROM taxi_trips t
        JOIN taxi_zone_lookup pickup ON t.PULocationID = pickup.LocationID
        JOIN taxi_zone_lookup dropoff ON t.DOLocationID = dropoff.LocationID
        WHERE trip_distance > 10
        GROUP BY 1, 2, 3, 4, 5, 6
        ORDER BY trip_count DESC
        LIMIT 10
    """)

def main():
    """
    Main function to run the taxi data processing pipeline
    """
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("NYC Taxi Data Processing") \
        .config("spark.sql.session.timeZone", "America/New_York") \
        .getOrCreate()
    
    # Set paths
    raw_data_path = "/tmp/taxi_data/raw"
    processed_data_path = "/tmp/taxi_data/processed"
    
    # Ensure directories exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    
    try:
        # Download data for specified year/month
        year = 2024
        month = 9  # September
        
        # Download taxi data
        taxi_df = download_taxi_data(spark, year, month, raw_data_path)
        if taxi_df is None:
            print("Failed to download taxi data. Exiting.")
            return
        
        # Download zone lookup data
        zone_df = download_zone_lookup(spark, raw_data_path)
        if zone_df is None:
            print("Failed to download zone lookup. Exiting.")
            return
        
        # Process the data
        print("Processing taxi data...")
        processed_df = process_taxi_data(taxi_df)
        processed_df = identify_outliers(processed_df)
        
        # Register as temp view for SQL queries
        processed_df.createOrReplaceTempView("taxi_trips")
        
        # Create aggregations
        print("Creating aggregations...")
        hourly_agg = create_hourly_aggregations(spark, processed_df)
        daily_agg = create_daily_aggregations(spark, processed_df)
        
        # Save aggregations
        hourly_agg.write.mode("overwrite").parquet(f"{processed_data_path}/hourly_agg")
        daily_agg.write.mode("overwrite").parquet(f"{processed_data_path}/daily_agg")
        
        # Run required analysis queries
        print("Running analysis queries...")
        top_locations = get_top_pickup_locations(spark, zone_df)
        weekday_patterns = get_weekday_weekend_patterns(spark)
        long_trips = get_long_trips(spark, zone_df)
        
        # Save analysis results
        top_locations.write.mode("overwrite").parquet(f"{processed_data_path}/top_locations")
        weekday_patterns.write.mode("overwrite").parquet(f"{processed_data_path}/weekday_patterns")
        long_trips.write.mode("overwrite").parquet(f"{processed_data_path}/long_trips")
        
        # Show results
        print("\nTop 10 pickup locations by revenue:")
        top_locations.show()
        
        print("\nWeekday vs Weekend patterns:")
        weekday_patterns.show()
        
        print("\nTop 10 Long trip routes:")
        long_trips.show()
        
        print("\nProcessing complete. Results saved to:", processed_data_path)
        
    except Exception as e:
        print(f"Error processing taxi data: {str(e)}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()