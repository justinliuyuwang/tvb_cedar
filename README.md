# tvb_cedar

1) https://github.com/ins-amu/virtual_aging_brain
clone and install this on cedar (go up to step 2 in the readme)

2) In a second directory, clone the showcase tutorial on to cedar (including the entire virtual_ageing directory)
https://lab.ch.ebrains.eu/user-redirect/lab/tree/shared/SGA3%20D1.2%20Showcase%201/virtual_ageing

You can also find the tutorial here:
https://drive.ebrains.eu/
and then go to shared with all groups > find your copied version of the collab for this showcase

3) allocate a node using `salloc --time=4:00:00 --mem=64000MB --account=\<account\>`

4) run `module load scipy-stack`

5) cd into the first repo you cloned in step 1 and run . env/bin/activate

6) cd into notebooks in the showcase tutorial you cloned in step 2 and run:
`pip install --upgrade pip`
`pip install -e ..`

7) you should be able to run the py script (~/projects/def-rmcintos/jwangbay/model1.py) with your own sc/weights file hardcoded in the appropriate lines (line 13 i think). currently it just produces a bold timeseries. you can copy what you need from the showcase as needed



