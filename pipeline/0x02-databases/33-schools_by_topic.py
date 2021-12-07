#!/usr/bin/env python3
"""Returns the list of school having a specific topic"""
import pymongo


def schools_by_topic(mongo_collection, topic):
    """
    List of school having a specific topic
    """
    list_schools = []
    mycol = mongo_collection.find({'topics': {'$all': [topic]}})
    for res in mycol:
        list_schools.append(res)

    return list_schools
