### A program to test the functionality of the longest possible substring without any repeating characters.
def longst_possible_substring(s):
    last_seen={}
    max_length=0
    left=0
    for right in range(len(s)):
        while s[right] in last_seen:
            s.remove(s[left])


        s.add(s[right])
        max_length=max(max_length,right-left+1)
        last_seen[s[right]]=right
    return max_length

s="abcabcbb"
print(longst_possible_substring(s))


