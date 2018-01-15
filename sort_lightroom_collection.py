import csv
import sqlite3
from path import Path

# this script sorts the images in one folder using the output of the NIMA scripts
# you need to have a csv file containing the results (RESULTS_CSV)
# and a collection in lightroom containing the images of the folder you
# analyzed

DEFAULT_CATALOG = r's:\Pictures\Lightroom\Lightroom Catalog.lrcat'
LIGHTROOM_COLLECTION_NAME = 'expert_A'
RESULTS_CSV = r's:\projects\neural-image-assessment\results.csv'


def open_catalog():
    catalog = DEFAULT_CATALOG
    db_conn = sqlite3.connect(catalog)
    db = db_conn.cursor()
    db.row_factory = sqlite3.Row
    return db


DB = open_catalog()


def get_collection_id(collection_name):
    collections_query = DB.execute(
        'select * from AgLibraryCollection where name=\'{}\''.format(
            collection_name))

    collection_sets = {}
    collections = {}

    for cxn in collections_query:
        if cxn['creationId'] == 'com.adobe.ag.library.group':
            path = cxn['genealogy']
            cmps = path.split('/')
            if len(cmps) == 2:
                # root level collection/set
                collection_sets[path] = cxn['name']
            else:
                parent_path = "/".join(cmps[:-1])
                parent_name = collection_sets[parent_path]
                current_name = parent_name + "/" + cxn['name']
                collection_sets[path] = current_name
        elif cxn['creationId'] == 'com.adobe.ag.library.collection':
            path = cxn['genealogy']
            cmps = path.split('/')
            parent_path = "/".join(cmps[:-1])
            if parent_path:
                parent_name = collection_sets[parent_path]
                current_name = parent_name + "/" + cxn['name']
            else:
                current_name = cxn['name']
            collection_id = cxn['id_local']
            collections[collection_id] = {'name': current_name}
    return collections


def get_new_file_order(csv_file):
    with open(csv_file, 'r', encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';', lineterminator='\n')
        header = next(csvreader)
        files = [str(Path(row[0]).basename()) for row in csvreader]
    files = dict([(fn, float(i) + 0.01) for i, fn in enumerate(files)])
    return files


def changeOrder(collection_name=LIGHTROOM_COLLECTION_NAME,
                csv_file=RESULTS_CSV):
    collections = get_collection_id(collection_name)
    assert (len(collections) == 1)
    id = list(collections.keys())[0]

    query = """
            select t1.id_local,t1.positionInCollection,t3.originalFilename
            from AgLibraryCollectionImage t1
        inner join Adobe_images t2
            ON t1.image=t2.id_local
        inner join AgLibraryFile t3
            ON t2.rootFile=t3.id_local
        where t1.collection='{}'
    """.format(id)
    results = DB.execute(query)

    id_local_to_fn = {}
    for row in results:
        id_local_to_fn[row['id_local']] = row['originalFilename']
    files = get_new_file_order(csv_file)

    next_pos = len(files) + 0.01
    update = []
    for id_local, fn in id_local_to_fn.items():
        if fn in files:
            update.append((id_local, files[fn]))
        else:
            update.append((id_local, next_pos))
            next_pos += 1.0

    for id_local, pos in update:
        query = """
          UPDATE AgLibraryCollectionImage SET positionInCollection={}
          WHERE id_local={}
        """.format(pos, id_local)
        DB.execute(query)
        print(query)

    # Save (commit) the changes
    DB.connection.commit()
    DB.connection.close()


changeOrder()
