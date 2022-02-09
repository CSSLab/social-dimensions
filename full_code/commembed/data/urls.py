# generates urls
import sys

base_url = "http://files.pushshift.io/reddit/"

def get_urls(typ, only_study_period=False):
    if typ == "submissions":

        pre_study_period = ["RS_v2_2005-%02d.xz" % (m) for m in range(6, 13)] + \
            ["RS_v2_%04d-%02d.xz" % (y, m) for y in range(2006, 2011) for m in range(1, 13)] + \
            ["RS_%04d-%02d.bz2" % (y, m) for y in range(2011, 2015) for m in range(1, 13)]

        study_period = ["RS_%04d-%02d.zst" % (y, m) for y in range(2015, 2017) for m in range(1, 13)] + \
            ["RS_2017-%02d.bz2" % (m) for m in range(1, 12)] + \
            ["RS_2017-12.xz"] + \
            ["RS_2018-%02d.xz" % (m) for m in range(1, 11)] + \
            ["RS_2018-11.zst", "RS_2018-12.zst"] + \
            ["RS_2019-%02d.zst" % (m) for m in range(1, 13)]
        
        for filename in (pre_study_period if not only_study_period else []) + study_period:
            yield((filename, base_url + "submissions/" + filename))

    elif typ == "comments":
        pre_study_period = ["RC_2005-12.bz2"] + \
            ["RC_%04d-%02d.bz2" % (y, m) for y in range(2006, 2015) for m in range(1, 13)]
        
        study_period = ["RC_%04d-%02d.bz2" % (y, m) for y in range(2015, 2017) for m in range(1, 13)] + \
            ["RC_2017-%02d.bz2" % (m) for m in range(1, 12)] + \
            ["RC_2017-12.xz"] + \
            ["RC_2018-%02d.xz" % (m) for m in range(1, 10)] + \
            ["RC_2018-10.zst", "RC_2018-11.zst", "RC_2018-12.zst"] + \
            ["RC_2019-%02d.zst" % (m) for m in range(1, 13)]

        for filename in (pre_study_period if not only_study_period else []) + study_period:
            yield((filename, base_url + "comments/" + filename))

    else:
        raise Exception("Incorrect type %s" % typ)
        
if __name__ == "__main__":
    print('\n'.join([url[1] for url in get_urls(sys.argv[1])]))