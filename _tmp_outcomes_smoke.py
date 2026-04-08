from weiss_rl.league.outcomes import OnlineOutcomeTracker
t=OnlineOutcomeTracker(window_size=3, draw_value=0.5)
oid="oppA"
for x in ["w","w","w"]:
    t.update(oid,x)
assert t.counts(oid)==(3,0,0)
t.update(oid,"l")
assert t.counts(oid)==(2,1,0)
print("ok")
