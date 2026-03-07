# src/features/dns_features.py
import pandas as pd
import numpy as np

# Approximate coordinates of India (NIC HQ)
INDIA_LAT, INDIA_LON = 20.5937, 78.9629

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    from math import radians, cos, sin, asin, sqrt
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def get_country_coords(country_code):
    """Approximate country coordinates (simplified)"""
    coords = {
        'IN': (20.5937, 78.9629),   # India
        'US': (37.0902, -95.7129),  # USA
        'GB': (55.3781, -3.4360),   # UK
        'DE': (51.1657, 10.4515),   # Germany
        'JP': (36.2048, 138.2529),  # Japan
        'CA': (56.1304, -106.3468), # Canada
        'AU': (-25.2744, 133.7751), # Australia
        'BR': (-14.2350, -51.9253), # Brazil
        'FR': (46.2276, 2.2137),    # France
        'RU': (61.5240, 105.3188),  # Russia
    }
    return coords.get(country_code.upper(), (0, 0))

def extract_dns_features(df):
    df = df.copy()
    
    # MX record presence (from your enriched data)
    df['has_mx_record'] = df['DNS Records (if any)'].notna() & (df['DNS Records (if any)'] != '')
    
    # Number of nameservers (simplified: count if present)
    df['num_nameservers'] = df['Name Servers'].fillna('').str.count(';') + 1
    df['num_nameservers'] = df['num_nameservers'].where(df['Name Servers'].notna(), 0)
    
    # Geo-distance to India (NIC)
    df['hosting_country_code'] = df['Hosting Country'].fillna('').str[:2].str.upper()
    df[['host_lat', 'host_lon']] = df['hosting_country_code'].apply(
        lambda cc: pd.Series(get_country_coords(cc))
    )
    df['ip_geodistance_to_cse'] = df.apply(
        lambda row: haversine_distance(INDIA_LAT, INDIA_LON, row['host_lat'], row['host_lon']),
        axis=1
    )
    
    return df[['has_mx_record', 'num_nameservers', 'ip_geodistance_to_cse']]