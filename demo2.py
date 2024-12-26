import json

def loadKeywords(job_title,experience):
    with open("JobProfileKeywords.json", "r") as f:
        # Normalize job title to match the keys in the JSON
        job_title_load = json.load(f)
        job_title_key = job_title.replace(" ", "_")
        
        # Retrieve the keywords for the specific job title and experience level
        experience_level = str(experience)
        # return job_title_key, experience
        return job_title_load.get(job_title_key).get(experience_level)

# Example usag
x= loadKeywords("Java developer",1)
print(x,"1111111111111",type(x))