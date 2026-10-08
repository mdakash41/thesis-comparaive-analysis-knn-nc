def twoSum(nums, target):
    num_hash = dict()
    f_index = s_index = None
    found = False
    for i, item in enumerate(nums):
        remaining = target - item
        s_index = remaining in num_hash
        if s_index:
            f_index = i
            s_index = num_hash.get(remaining)
            found = True
            break
        else:
            num_hash[item] = i
    if found:
        return sorted([f_index, s_index])
    else:
        return []

if __name__ == '__main__':
    print(twoSum(**{"nums":[2,7,11,15],"target":9}))
