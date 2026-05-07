#HU dictionary

# Values pulled from : https://radiopaedia.org/articles/hounsfield-unit?lang=us
#                      https://www.uvm.edu/~rchawla/series_home/ct/ct3.html
#                      https://collectiveminds.health/articles/understanding-hounsfield-units-hu-the-complete-guide-to-ct-numbers-and-density-values
#                      https://en.wikipedia.org/wiki/Hounsfield_scale

keys = ["air", "bone_gen","bone_cort", "bone_trab", "brain_gray", "brain_white", "subq_fat","liver","lungs","metal",
        "muscle","renal_cort","spleen","water", "soft_tissue","fat","lung","fst"]
upper = [-1000,800,1900,800,40,30,-100,50,-650,4000,50,30,45,0,300,-90,-500,200]
lower = [-1000,300,500,300,40,30,-115,45,-950,3001,45,25,40,0,-150,-120,-500,-250]

HU_dict = {k: (v1, v2) for k, v1, v2 in zip(keys, upper, lower)}

print(HU_dict['muscle'])

