
record = {}

def memo_search(a):
	if min(a[0], a[1], a[2], a[3]) == 0:
		return 0
	try:
		x = record[a]
		return x
	except:
		record[a] = \
			0.25 * (memo_search((a[0], a[1], a[2], a[3] - 1)) + memo_search((a[0], a[1], a[2] - 1, a[3])) +
			 memo_search((a[0], a[1] - 1, a[2], a[3])) + memo_search((a[0] - 1, a[1], a[2], a[3]))) + 1
		return record[a]


def memo_search2(a):
	if min(a[0], a[1]) == 0:
		return 0
	try:
		x = record[a]
		return x
	except:
		record[a] = \
			0.5 * (memo_search2((a[0], a[1] - 1)) + memo_search2((a[0] - 1, a[1]))) + 1
		return record[a]

print(memo_search((4, 4, 4, 4)))
