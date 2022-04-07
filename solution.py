import pandas
import glob

def main():
    demand = glob.glob("demand/*.csv")
    print(demand)

if __name__ =="__main__":
    main()