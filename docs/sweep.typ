#import "@preview/ctheorems:1.1.3": *
#import "@preview/cetz:0.3.1"
#import "@preview/lovelace:0.3.0": *
#set par(justify: true)

#show: thmrules
#let lemma = thmbox("lemma", "Lemma")
#let def = thmbox("definition", "Definition")
#let proof = thmproof("proof", "Proof")

#let inexact(term) = {
  block(inset: 16pt,
    block(
      inset: 16pt,
      stroke: 1pt + gray,
      radius: 12pt,
      text(
        size: 10pt,
        [Note for inexact arithmetic: #term]
      )
    )
  )
}

// TODO: figure out how to get rid of the ":" in the caption
#let invariant(term) = {
figure(
  block(inset: 16pt,
      text(
        size: 10pt,
        [#term]
    )
  ),
  kind: "invariant",
  supplement: "invariant",
  caption: ""
)
}

We'll be talking about sweep line algorithms, where the sweep line is horizontal and increasing in $y$.
Therefore, every line segment "starts" at the coordinate with smaller $y$ and "ends" at the coordinate
with larger $y$ (we'll assume for now that there are no horizontal segments). We'll parametrize each
line segment as a function of $y$. So if $alpha: [y_0 (alpha), y_1 (alpha)] arrow bb(R)$ is a line segment
then $alpha(y)$ is the $x$ coordinate at $y$-coordinate $y in [y_0 (alpha), y_1 (alpha)]$.
We write $theta_alpha$ for the angle that $alpha$ makes with the positive horizontal axis.
Let's have a picture. (In the discussion, it won't matter whether positive $y$ points up or down, but in the
pictures we'll adopt the graphics convention of having positive $y$ point down.)

#cetz.canvas({
  import cetz.draw: *

  line((1, 3), (0, 0), name: "a")
  line((-4, 3), (4, 3), stroke: (dash: "dotted"))
  line((-4, 0), (4, 0), stroke: (dash: "dotted"))
  content((-4, 3), $y_0(alpha)$, anchor: "east")
  content((-4, 0), $y_1(alpha)$, anchor: "east")
  content((0.6, 1.5), $alpha$, anchor: "west")

  cetz.angle.angle("a.start", "a.end", (4, 3), label: $theta_alpha$, label-radius: 0.8)
})

We'll be dealing with inexact arithmetic, so let's define some "error bars" on our line segments.
For an error parameter $epsilon > 0$, offsetting from $alpha$ by $plus.minus epsilon$ in the perpendicular-to-$alpha$ direction
is the same as offsetting by $alpha plus.minus epsilon / (|cos theta_alpha|)$ in the horizontal direction.
Roughly speaking, the "error bars" on $alpha$ amount to adding this horizontal error. But we'll be slightly
more accurate around the corners, by truncating these error bars to the horizontal extends of $alpha$. Precisely, we define

$
alpha_(+,epsilon)(y) = min(alpha(y) + epsilon / (|cos theta_alpha|), max(alpha(y_0), alpha(y_1))) \
alpha_(-,epsilon)(y) = max(alpha(y) - epsilon / (|cos theta_alpha|), min(alpha(y_0), alpha(y_1))) \
$

In pictures, the gray shaded region is the region between $alpha_(-,epsilon)$ and $alpha_(+,epsilon)$:

#cetz.canvas({
  import cetz.draw: *

  line((0.5, 3), (0, 1.5), (0, 0), (0.5, 0), (1, 1.5), (1, 3), close: true, fill: gray, stroke: 0pt)
  line((0.5, 3), (0, 1.5), (0, 0), stroke: ( dash: "dashed" ))
  line((0.5, 0), (1, 1.5), (1, 3), stroke: ( dash: "dashed" ))

  line((1, 3), (0, 0), name: "a")
  line((-4, 3), (4, 3), stroke: (dash: "dotted"))
  line((-4, 0), (4, 0), stroke: (dash: "dotted"))
  content((-4, 3), $y_0(alpha)$, anchor: "east")
  content((-4, 0), $y_1(alpha)$, anchor: "east")
  content((0.6, 1.5), $alpha$, anchor: "west")

  content((0.8, 0.5), $alpha_(+,epsilon)$, anchor: "west")
  content((0.2, 2.4), $alpha_(-,epsilon)$, anchor: "east")
})


Define a relation $prec_(y,epsilon)$ on line segments whose domain contains $y$, where
$alpha prec_(y,epsilon) beta$ if $alpha_(+,epsilon)(y) < beta_(-,epsilon)(y)$.
Intuitively, $alpha prec_(y,epsilon) beta$ if $alpha$ is definitely to the left of $beta$
at height $y$: $alpha$ is far enough to the left that their error bars don't overlap.
Clearly, for a given $y$ and $epsilon$ there are three mutually exclusive possibilities: either
$alpha prec_(y,epsilon) beta$ or $beta prec_(y,epsilon) alpha$ or neither of the two holds. We'll denote
this third possibility by $alpha approx_(y,epsilon) beta$, and we write
$alpha prec.tilde_(y,epsilon) beta$ for "$alpha prec_(y,epsilon) beta$ or $alpha approx_(y,epsilon) beta$."

Here are a few basic properties of our definitions:
#lemma[
1. For any $y$ and any $epsilon > 0$, $prec_(y,epsilon)$ is transitive:
  if $alpha prec_(y,epsilon) beta$
  and $beta prec_(y,epsilon) gamma$ then $alpha prec_(y,epsilon) gamma$. (However, $prec.tilde_(y,epsilon)$ is not transitive.)
2. For any $y$ and any $epsilon > 0$,
  if $alpha prec_(y,epsilon) beta$
  and $beta prec.tilde_(y,epsilon) gamma$ then $alpha prec.tilde_(y,epsilon) gamma$.
3. For any $y$ and any $epsilon > 0$,
  if $alpha prec.tilde_(y,epsilon) beta$
  and $beta prec_(y,epsilon) gamma$ then $alpha prec.tilde_(y,epsilon) gamma$.
4. For any $y$, the relation $prec_(y,epsilon)$ is monotone in $epsilon$, in that if $alpha prec_(y,epsilon) beta$ then $alpha prec_(y,eta) beta$ for
  any $eta in (0, epsilon)$.
]

#def[
  Suppose $alpha$ and $beta$ are two segments whose domain contains $y$. We say that *$alpha$ and $beta$
  are $epsilon$-close from $y$ onwards* if
  $alpha approx_(z,epsilon) beta$
  for all $y <= z <= min(y_1(alpha), y_1(beta))$.
] <close_from_y_onwards>

#def[
  Suppose $alpha$ and $beta$ are two segments whose domain contains $y$. We say that *$(alpha, beta)$
  $epsilon$-cross by $y$* if $y$ belongs to both domains and $alpha succ_(y,epsilon) beta$.
  We say that *$(alpha, beta)$ $epsilon$-cross* if they $epsilon$-cross by $min(y_1 (alpha), y_1 (beta))$.
]

Note that the definition of $epsilon$-crossing is not symmetric: $(alpha, beta)$ $epsilon$-crossing is
not the same as $(beta, alpha)$ $epsilon$-crossing. We will usually talk about $(alpha, beta)$ $epsilon$-crossing
in the context that $alpha$ starts off to the left of $beta$, and in this context "$(alpha, beta)$ $epsilon$-cross" means
that at some height before the end of $alpha$ and $beta$, $alpha$ has definitely crossed to the right of $beta$.

== Partially ordered sweep-lines

At our first pass, we won't try to detect intersections at all. Instead, we'll produce
a continuum of sweep-lines (constant except at a finite number of points) that *approximately*
track the horizontal order of the segments.

#def[
The ordered collection $(alpha^1, ..., alpha^m)$ of line segments is #emph[$epsilon$-ordered at $y$]
if each $alpha^i$ has $y$ in its domain and $alpha^i prec.tilde_(y,epsilon) alpha^j$ for all $1 <= i < j <= m$.
]

Our algorithm will produce a family of sweep-lines that are $epsilon$-ordered at every $y$
(and also the sweep-lines will be #emph[complete] in the sense that the sweep-line at $y$ will contain all line segments whose
domain contains $y$). This seems weaker than finding all the intersections (for example, because if you
find all intersections you can use them to produce a completely ordered family of sweep-lines), but
in fact they're more-or-less equivalent: given weakly-ordered sweep-lines, you can perturb the lines so
that your previously-weak order becomes a real order:

#lemma[
If $(alpha^1, ..., alpha^m)$ is $epsilon$-ordered at $y$ then there exist $x^1 <= ... <= x^m$ such that
$alpha_(-,epsilon)^i (y) <= x^i <= alpha_(+,epsilon)^i(y)$ for all $i$.
]

#proof[
Define $x^i = max_(j <= i) alpha_(+,epsilon)^j(y)$.
]

So once we have our family of approximate sweep-lines, we can go back and perturb the lines so that our
approximate ordering is exact.

One consequence of our approximate approach is that we need to do a little extra bookkeeping to maintain
the continuity of the input paths: when one segment exits and its path-neighbor enters, we need to remember
that they are connected because the approximate sweep-line might not keep them together. We'll ignore this
bookkeeping for now;
the goal here is to get into detail about the sweep-line invariants, and prove that we can maintain them.

== The sweep-line invariants

We're going to have a sweep-line that depends on $y$. When we need to emphasize this, we'll use the
cumbersome but explicit notation
$(alpha_y^1, ..., alpha_y^(m_y))$.
In addition to the sweep-line, we'll maintain a queue of "events." The events are ordered by $y$-position
and they are similar to the classical Bentley-Ottmann events so we'll skimp on the details here. There
is an "enter" event, an "exit" event, and an "intersection" event.

Our sweep-line will maintain two invariants:
+ At every $y$, the sweep-line is $epsilon$-ordered at $y$. (We'll call this the "order" invariant.)
+ For every $y$ and every $1 <= i < j <= m_y$, if $alpha_y^i$ and $alpha_y^j$ $epsilon$-cross
  then the event queue contains an event between $i$ and $j$,
  and at least one of these events occurs before the $epsilon$-crossing height, or at $y$.
  (We'll call this the "crossing" invariant.)

(When we say the event queue contains an event between $i$ and $j$, we mean that either there's
an exit event for some $alpha^k$ with $i < k < j$ or there's an intersection event for some $alpha^k$ and $alpha^ell$
with $i <= k < ell <= j$.)

Hopefully the first invariant is already well-motivated, so let's discuss the second.
To naively ensure that we find
all intersections, we could adopt a stronger crossing invariant that requires all pairs of intersecting segments
in the sweep-line to immediately put an intersection event in the queue. This would be inefficient to maintain because
it would require testing all pairs of segments. The main Bentley-Ottmann observation is that it's enough to have intersection
events for all sweep-line-adjacent pairs, because any pair of lines will have to become sweep-line adjacent strictly
before they intersect. We can't rely only on sweep-line adjacency because of the $epsilon$ fudge, but our "crossing"
event essentially encodes the same property. Note that if $j = i+1$ and $y$ is just before the $epsilon$-crossing height and
there are no other segments nearby, then the mysterious $j'$ must be either $i$ or $j$ (because there is nothing in between)
and the mysterious queue event must be the crossing of $i$ and $j$ (it couldn't be an exit event, because we assumed they
cross and they must cross before they exit). In other cases, the crossing event ensures that even if we haven't
recorded the upcoming crossing of $alpha^i$ and $alpha^j$, something will happen in between them that will give us
a chance to test their crossing.

== Sweeping the sweep line

The first observation is that the sweep line invariants are maintained as we increase $y$ continuously up
to the next event:
+ For the order invariant, first note that there is an event whenever $y$ leaves the domain of a segment, so $y$ remains in all domains until the next event.
  Moreover, if at any $y' > y$ the ordering breaks, two line segments must have $epsilon$-crossed one another by $y'$.
  The third invariant guarantees that there's an event before this happens, so by the contra-positive until an event happens the ordering
  constraint is maintained.
+ The crossing invariant is maintained because the set of things to check (i.e. the set of line segments that $epsilon$-cross
  one another after $y$) only shrinks as $y$ increases.

== Interaction with adjacent segments

In vanilla Bentley-Ottmann, each segment gets compared to its two sweep-line neighbors; they can either intersect or not intersect.
When numerical errors are taken into account, we may need to compare to
more segments. TODO: draw an example with three segments.

First of all, there are two kinds of intersections: a "clear" intersection where we can compute the
`y` coordinate of the crossing and it's definitely larger than the current sweep line; in this case we insert an intersection event into the queue.
On the other hand, it could be that
the two lines definitely intersect somewhere but they're too close to coincident to really figure out when, or maybe their
intersection point is just about at the current sweep line. In this case -- which we call a "shuffle" intersection --
we re-order the current sweep line.

Here's a definition:
#def[
Suppose $alpha prec_(y,epsilon) beta$.
We say that *$(alpha, beta)$ have a shuffle intersection at $y$* if $beta(y) < alpha(y)$ and $(alpha, beta)$ $epsilon$-cross.
We say that *$(alpha, beta)$ have a clear intersection at $y$* if $(alpha, beta)$ $epsilon$-cross but do not have a shuffle intersection at $y$.
]

Fix $y$ and suppose we have two segments $alpha$ and $beta$ satisfying $alpha prec.tilde_(y,epsilon) beta$.
- If $(alpha, beta)$ $epsilon$-cross then they have a clear intersection or a shuffle intersection.
- If $alpha$ and $beta$ clearly intersect, there is an algorithm that computes a lower bound $hat(y)$ on the exact intersection height,
  with the property that $alpha succ.tilde_(hat(y),epsilon) beta$.

TODO: the second of these properties should be an assumption (on $epsilon$, and on the admissible segments $alpha$ and $beta$). I think
this can be done for every pair that intersect, not just "clearly" intersecting ones.


Clearly these definitions satisfy the first desired property (if $(alpha, beta)$ $epsilon$-cross then there is some kind
of intersection). To check that they satisfy the second property requires some more careful analysis (which will only
work when $epsilon$ is sufficiently large depending on the precision of the floating point numbers we're using),
but the point is that if $(alpha, beta)$ clearly cross then their "slopes" (not exactly their slopes, but close enough)
are at least $epsilon$ apart and so we can compute intersections accurately enough.

#inexact[
in practice, there will be some numerical
errors in evaluating these definitions, but that's ok if we're careful about the error direction: if $beta(y)$ is just barely bigger than $alpha(y)$, we
can allow a false positive in the shuffle intersection test, and the numerical stability for the clear intersection
test will just be a little bit worse. On the other hand, we *insist* that for a clear intersection,
$beta(y)$ is truly at least $alpha(y)$ (and so we can give a lower bound on the intersection height that's larger than the current sweep height).
]

Now we're ready to talk about our replacement for Bentley-Ottmann's "just compare to the next segment" step. Imagine
we have an $epsilon$-ordered collection $(alpha^1, ..., alpha^m)$ and we want to see if $alpha^i$ intersects with anything
"before" it in the collection. We'll call this procedure an *intersection scan to the left*.
For each $j$ from $i - 1$ down to $1$, we test $alpha^j$ and $alpha^i$ for intersections.
- If there's a shuffle intersection, we add an intersection event to the event queue (with $y$-coordinate at the current scan-line)
  and continue.
- If there's a clear intersection, we add an intersection event to the event queue (with $y$-coordinate that lower-bounds the
  true intersection height; note that the definition of a clear intersection implies that we can find such a lower bound that's
  at least the current scan-line's height). Then we stop scanning.
- If we encounter a segment that's strictly to the left of $alpha^i$ from $y$ onwards, we stop scanning.
Just to compare once more to Bentley-Ottmann: with exact computations you only need to compare to one segment. Either
it intersects or it doesn't. With inexact computations, you keep scanning left until you either find something that
definitely intersects or definitely doesn't.

The *intersection scan to the right* is very similar to the intersection scan to the left: just increase $j$ instead of
decreasing it. An *intersection scan* combines the two (in either order).

#lemma[
Suppose $(alpha^1, ..., alpha^(i-1))$ satisfies the ordering and crossing invariants at $y$, and suppose that
$(alpha^1, ..., alpha^i)$ satisfies the ordering invariant at $y$. After running an intersection scan to the left,
it also satisfies the crossing invariant at $y$.

Similarly, suppose $(alpha^(i+1), ..., alpha^m)$ satisfies the ordering and crossing invariants at $y$, and suppose that
$(alpha^i, ..., alpha^m)$ satisfies the ordering invariant at $y$. After running an intersection scan to the right,
it also satisfies the crossing invariant at $y$.
]<lem-intersection-scan>

#proof[
We'll prove the first claim; the second is similar. Because the crossing invariant holds without $alpha^i$, we only
need to consider intersections involving $alpha^i$. So suppose that $j < i$ and $(alpha^j, alpha^i)$ $epsilon$-cross.
If we encountered $alpha^j$ when scanning for $alpha^i$'s intersections, we would have inserted an intersection event
and that intersection event would witness the crossing invariant. So let's assume that when scanning for $alpha^i$'s clear intersections,
we stopped the scan at $alpha^k$ for some $k > j$.

There are two reasons we might have stopped the clear intersection scan:
- if $alpha^k$ clearly intersects $alpha^i$, we inserted an intersection event at some $y$-coordinate that's less than or equal to the
  true intersection $y$-coordinate. Let $y_k$ be the $y$-coordinate of the intersection event that we inserted. There are two sub-cases:
  - if $alpha^j$ $epsilon$-crosses $alpha^i$ after $y_k$, then the intersection event we inserted witnesses
    the crossing invariant.
  - otherwise, $alpha^j$ $epsilon$-crosses $alpha^i$ before $y_k$, and therefore also before the true intersection between
    $alpha^k$ and $alpha^i$. Therefore, $alpha^j$ $epsilon$-crosses $alpha^k$ before $alpha^j$ $epsilon$-crosses $alpha^i$. By the
    crossing invariant for the old sweep-line, the queue contains some witnessing event for the $epsilon$-crossing of $alpha^j$ and $alpha^k$;
    and this event also witnesses the crossing invariant for the $epsilon$-crossing between $alpha^j$ and $alpha^i$.
- Otherwise, scanning terminated because $alpha^k$ doesn't meet $alpha^i$.
  - If $alpha^j$ $epsilon$-crosses $alpha^i$ after $alpha^k$ ends, the exit event witnesses the crossing invariant.
  - Otherwise, $alpha^j$ must $epsilon$-cross $alpha^k$ before it $epsilon$-crosses $alpha^i$. The crossing invariant
    for the old sweep-line guarantees the existence of an event witnessing the new sweep-line invariant.
]

#inexact[
when deciding when to stop scanning to the left, we can't exactly check whether a segment is strictly to the left of $alpha^i$.
We allow false negatives but not false positives; this might result in a little extra scanning, but it ensures that the proof
still works.
]

== An "enter" event

When inserting a new segment into the current sweep-line, we first choose its sweep-line position using
a binary search on its horizontal coordinate. Let's write $(alpha^1, dots, alpha^m)$ for the sweep-line
before inserting the new segment, and let's call the new segment $beta$. First, we observe that
there is a place to insert the new segment while preserving the ordering invariant.

#lemma[
Suppose $(alpha^1, ..., alpha^m)$ is $epsilon$-ordered at $y$.
If $alpha^i (y) <= beta(y) <= alpha^(i+1) (y)$ then
$(alpha^1, ..., alpha^i, beta, alpha^(i+1), ..., alpha^m)$ are $epsilon$-ordered at $y$.
(Here, we can allow the corner cases $i = 0$ and $i = m$ by declaring that
"$alpha^0(y)$" means $-infinity$ and "$alpha^(m+1) (y)$" means $infinity$).
]<lem-insert-preserving-order>

#proof[
Since $(alpha^1, ..., alpha^m)$ was $epsilon$-ordered at $y$, it suffices to compare beta with all $alpha^j$.
So now fix $1 <= j <= m$. If $j >= i + 1$ then
$beta (y) <= alpha^(i+1) (y) <= alpha^j (y) + epsilon$ (where the last inequality follows because $(alpha^1, ..., alpha^m)$ is $epsilon$-ordered.
On the other hand, if $j <= i$ then $alpha^j (y) - epsilon <= alpha^i (y) <= beta (y)$.
]

#inexact[
the inequalities $alpha^i (y) <= beta(y) <= alpha^(i+1) (y)$ need to be exact for the proof to work.
In the face of inexact computations, the problematic case is when $beta(y)$ is very close to (say) $alpha^i (y)$
and there is some $j < i$ with $alpha^j (y)$ very close to $alpha^i(y) + epsilon$. In this case we might
wrongly think that $beta(y) >= alpha^i (y)$, and by inserting $beta$ after $alpha^i (y)$ we break
the ordering invariant for $alpha^j$ and $beta$. This case can probably be handled by scanning left from $alpha^i$
to look for a problematic $alpha^j$; if we find one, ensure that $beta$ goes to the left of it. The correctness arguments
here require a bit more thought, but I think one important point is that we can't have a "problematic $alpha^j$" in *both*
directions, because if we did then those two problematic segments would be almost $2 epsilon$ out-of-order.
]

@lem-insert-preserving-order implies that we can insert a new segment while preserving the ordering invariant. By
@lem-intersection-scan, running an intersection scan restores the crossing invariant.
Thus, we can insert a new segment while preserving the sweep-line invariants.

== An "exit" event

When a segments exits the sweep-line, the ordering invariant clearly doesn't break.
Regarding the crossing invariant, it can only break because of $epsilon$-crossing pairs whose
crossing invariant was witnessed by the exit event that was just processed.
To restore the crossing invariant, we need to enqueue some new intersection events.

Let $(alpha^1, ..., alpha^m)$ be the sweep-line after removing the just-exited segment
which, we assume, used to live between $alpha^i$ and $alpha^(i+1)$. It is tempting to
intersection-scan $alpha^i$ to the right and $alpha^(i+1)$ to the left. By @lem-intersection-scan,
this ensures that the both $(alpha^1, ..., alpha^(i+1))$ and $(alpha^i, ..., alpha^m)$ satisfy
the crossing invariant. However, this does not imply that the whole sweep-line satisfies the
crossing invariant. For example, it is possible that $alpha^(i-1)$ sticks out $2/3 epsilon$
to the right of $alpha^i$ and $alpha^(i+2)$ sticks out $2/3 epsilon$ to the left of $alpha^(i+1)$,
and $alpha^(i-1)$ $epsilon$-crosses $alpha^(i+2)$ in violation of the crossing invariant.

To fix this problem, we will need a *strict intersection scan*, which is similar to the previous scan except
that we declare crossings at a threshold of $epsilon/2$ instead of $epsilon$, and
the stopping criterion is stricter. We'll describe only the leftward version,
starting at $alpha^i$. As before, we iterate $j$ from $i-1$ down to $1$. Unlike the non-strict version,
we will keep track of the soonest (i.e. smallest $y$-coordinate) clear intersection we've seen so far; call it $y^*$.
- If we encounter an $epsilon/2$-shuffle intersection, we add an intersection event, update $y^*$, and continue scanning.
- If we encounter as $epsilon/2$-clear intersection, we add an intersection event as before; we also update $y^*$ if appropriate.
  Then we check whether $alpha^j$ is at least $epsilon/2$ to the left of $alpha^j$ for all heights between $y$ and $y^*$.
  If so, we stop scanning.
- If we encounter a segment that's at least $epsilon/2$ to the left of $alpha^j$ for all heights between $y$ and $y^*$, we stop scanning.

The point of this strict intersection scan is that it imposes a crossing invariant with half the error (although only for $alpha^i$, and
only in one direction).
  
#lemma[
Suppose $(alpha^1, ..., alpha^(i-1))$ satisfies the ordering and crossing invariants at $y$, and suppose that
$(alpha^1, ..., alpha^i)$ satisfies the ordering invariant at $y$. After running a strict intersection scan to the left,
the following stronger crossing invariant holds: for every $j < i$, if $alpha^j$ $epsilon/2$-crosses $alpha^i$ then the
event queue contains an event between $j$ and $i$,
and at least one of these events occurs either before the $epsilon/2$-crossing height or at $y$.
]<lem-strict-intersection-scan>

#proof[
  Suppose $j < i$ and $alpha^j$ $epsilon/2$-crosses $alpha^i$. If the strict intersection scan included $j$, it would have
  inserted an intersection event for $alpha^j$ and $alpha^i$ (which would witness the crossing invariant),
  so let's suppose that the scan stopped at $k > j$ and recall the definition of $y^*$ (particularly the fact
  that we inserted an intersection event at height $y^*$). We consider two cases:
    - If $alpha^j$ $epsilon/2$-crosses $alpha^i$ after $y^*$ then the intersection event at $y^*$ witnesses the crossing invariant.
    - Otherwise, $alpha^j$ $epsilon/2$-crosses $alpha^i$ before $y^*$. Since $alpha^k$ is at least $epsilon/2$ less to the left
      of $alpha^i$ up to $y^*$, it follows that $alpha^j$ $epsilon$-crosses $alpha^k$ before it $epsilon/2$-crosses $alpha^i$.
      By the old crossing invariant, there is an event witnessing $alpha^j$ $epsilon$-crossing $alpha^k$; this same
      event witnesses $alpha^j$ $epsilon/2$-crossing $alpha^i$.
]

To finish handling the exit event, we run a strict intersection scan to the left of $alpha^i$ and a strict
intersection scan to the right of $alpha^(i+1)$. These two extra intersection scans ensure that without a witnessing
event in the queue, nothing "before" $alpha^i$ pokes out more than $epsilon/2$ to its right, and nothing "after"
$alpha^(i+1)$ pokes out more than $epsilon/2$ to its left. This eliminates the possibility of an unhandled
$epsilon$-crossing, and ensures that we maintain the crossing invariant.

== An "intersection" event

I don't think we've asserted this yet, but all intersection events in the queue always occur at a height where
the two crossing segments are within $epsilon$ of one another. Also, intersection events are only inserted
for $epsilon$-crossing segments, so it's always clear which order the segments should be in after the intersection.

Suppose our sweep-line is $(alpha^1, ..., alpha^m)$ and we've just encountered an intersection event for $alpha^i$ and $alpha^j$,
where $i < j$. First, we check whether $alpha^i$ and $alpha^j$ actually need to be swapped, as some other intersection event might
have already swapped them. If they're already in the correct order, we skip the re-ordering step and move on to the crossing invariant.
If $alpha^i$ and $alpha^j$
need to be re-ordered, we consider two possibilities: we can move $alpha^j$ before $alpha^i$ or we can move $alpha^i$ after $alpha^j$.
If $alpha^i(y) >= alpha^j(y)$ then either choice maintains the ordering invariant:

#lemma[
Suppose $(alpha^1, ..., alpha^m)$ satisfies the ordering invariant at $y$ and suppose that $i < j$ and $alpha^i (y) >= alpha^j (y)$.
Then $(alpha^1, ..., alpha^(i-1), alpha^(i+1), ..., alpha^j, alpha^i, alpha^(j+1), ... alpha^m)$ and
$(alpha^1, ..., alpha^(i-1), alpha^j, alpha^i, ..., alpha^(j-1), alpha^(j+1), ... alpha^m)$ both satisfy the ordering invariant
at $y$.
]<lem-reorder>

#proof[
  The two claims are essentially the same, so we will consider only the first one. When putting $alpha^i$ after $alpha^j$,
  the only pairs of segments whose order has changed are $(alpha^i, alpha^k)$ pairs for $i < k <= j$; hence, we only need
  to re-check the ordering invariant for those pairs. But for such $k$, the old ordering invariant implies
  that $alpha^k(y) - epsilon <= alpha^j (y)$, and since we assumed $alpha^j (y) <= alpha^i (y)$ it follows that
  $alpha^k (y) - epsilon <= alpha^i (y)$. Therefore, $alpha^i$ and $alpha^k$ satisfy the new ordering invariant.
]

It is possible, however that $alpha^i (y) < alpha^j (y)$. This is likely, even, because we tend to create intersection events
with a height that lower-bounds the actual intersection height. So in general we have a slightly more complex procedure:
choose $k$ with $i < k <= j$ to be the largest possible such that $alpha^k (y) <= alpha^i (y)$. Then move $alpha^i$ after $alpha^k$.
(This preserves the ordering invariant by @lem-reorder.) Finally, move $alpha^j$ before $alpha^i$'s new position. This maintains
the ordering invariant because the only pairs whose orders change were pairs of the form $(alpha^ell, alpha^j)$ for $k <= ell < j$
and the definition of $k$ ensures that $alpha^ell (y) >= alpha^i (y)$ (which is guaranteed to be at least $alpha^j (y) - epsilon$
by the fact that there was an intersection event).

=== Maintaining the crossing invariant

The final thing we need to do is explain how we maintain the crossing invariant. We'll do this in a slightly abstracted situation:
we have a sweep-line $(alpha^1, ..., alpha^m)$ that satisfies both invariants, and we move $alpha^i$ to after $alpha^j$ in such
a way that the ordering invariant is maintained. How do we maintain the crossing invariant? (The intersection-event processing
contains up to two of these "moves," so if we describe how to maintain the crossing invariant after one of these moves, we'll just
have to repeat that.)

In order to handle the crossing invariant of one swap, we treat it as an "exit" followed by an "entrance." That is,
to handle the removal of $alpha^i$ from its old location we run intersection scans on $alpha^(i-1)$ (strict to the left; non-strict
to the right) and $alpha^(i+1)$ (strict to the right; non-strict to the left). Then we run a (non-strict) intersection scan on $alpha^i$
in its new location.

=== The "push"

TODO: This should go in an earlier section at some point.

The following diagram shows an $epsilon$-ordered sweep line $(alpha, beta, gamma)$. Note that $beta prec_(y,epsilon) gamma$, but all
the other pairs are non-strictly ordered. Because our arithmetic isn't exact, we might encounter this sweep line at the intersection
event of $alpha$ and $gamma$ even though $alpha$ and $gamma$ don't intersect until a little bit later. But anyway, it's
reasonable to see this situation and think that we should swap the order of $alpha$ and $gamma$. We need to be careful, though: changing
the order to $(gamma, alpha, beta)$ doesn't work because $beta prec_(y,epsilon) gamma$. TODO: to explain this better I think
we need 4 lines...

#cetz.canvas({
  import cetz.draw: *

  content((-6, 2), $alpha$, anchor: "south", padding: 2pt)
  line((-7, 2), (3, -2), stroke: (dash: "dashed", thickness: 0.2pt))
  line((-5, 2), (5, -2), stroke: (dash: "dashed", thickness: 0.2pt))
  line((-6, 2), (4, -2), name: "a")

  content((5, 2), $gamma$, anchor: "south", padding: 2pt)
  line((6, 2), (-3, -2), stroke: (dash: "dashed", thickness: 0.2pt))
  line((4, 2), (-5, -2), stroke: (dash: "dashed", thickness: 0.2pt))
  line((5, 2), (-4, -2), name: "a")

  content((-1, 2), $beta$, anchor: "south", padding: 2pt)
  line((-1.3, 2), (-1.3, -2), stroke: (dash: "dashed", thickness: 0.2pt))
  line((-1, 2), (-1, -2))
  line((-0.7, 2), (-0.7, -2), stroke: (dash: "dashed", thickness: 0.2pt))

  line((-5, 0.05), (5, 0.05), stroke: (dash: "dotted", thickness: 0.5pt))
  content((-5, 0.05), $y$, anchor: "east", padding: 2pt)
})

=== Intersection scans

TODO: this is a re-write of an earlier section, and should get merged there

Suppose we have a collection of lines $(alpha^1, ..., alpha^n)$ that satisfy the ordering invariant
at $y$, and suppose that $(alpha^1, ..., alpha^(n-1))$ satisfy the crossing invariant at $y$.
To make the whole collection satisfy both invariants, we run the following algorithm.
We call this an *intersection scan to the left*.

#pseudocode-list[
  + $y^n <- y_1(alpha^n)$
  + *for* $i = n-1$ down to $1$
    + *if* $(alpha^i, alpha^n)$ $epsilon$-cross by $y^(i+1)$
      + let $z in [y, y^*]$ be a valid crossing height for $(alpha^i, alpha^n)$
      + insert an intersection event for $(alpha^i, alpha^n)$ at $z$
      + $y^i <- z$
    + #line-label(<shadow>) *else if* $alpha^i_+(z) <= alpha^n_+(z)$ for all $z in [y, y^(i+1)]$
      + *break*
    + *else*
      + $y^i <- y^(i+1)$
]

#lemma[
Suppose $(alpha^1, ..., alpha^n)$ satisfies the ordering invariants at $y$, and suppose that
$(alpha^1, ..., alpha^(n-1))$ satisfies the crossing invariant at $y$. After running an intersection scan to the left,
$(alpha^1, ..., alpha^n)$ satisfy the crossing invariant at $y$.
]<lem-intersection-scan2>

#proof[
  Let's first note a straightforward invariant of the algorithm:
  
  #invariant[
    at iteration $i$ of the loop, there is an event between $i$ and $n$ with height at most $y^(i+1)$
  ]<easy-loop-invariant>

  This invariant holds initially for $i = n-1$,
  witnessed by $alpha^n$'s exit event. Then we either copy $y^i <- y^(i+1)$ (and so the same event
  will witness the next iteration's invariant) or we set $y^i <- z$ and add a new intersection event that
  will witness the next iteration's invariant.

  We only need to check the crossing invariant for pairs of the form $(alpha^j, alpha^n)$. So take some $j < n$ and assume
  that $(alpha^j, alpha^n)$ $epsilon$-cross. We consider two cases:

  - if the loop in the intersection scan included $i = j$, then we break into two more cases:
    - if $(alpha^j, alpha^n)$ $epsilon$-cross by $y^(j+1)$ then the algorithm inserts an intersection event,
      and this new intersection event witnesses the crossing invariant
    - otherwise, by @easy-loop-invariant there is some other event that witnesses the crossing invariant
  - if the loop in the intersection scan didn't include $i = j$, then the loop terminated at some $i > j$;
    fix that $i$. If $(alpha^j, alpha^n)$ don't $epsilon$-cross by $y^(i+1)$ then @easy-loop-invariant gives
    us an event that witnesses the crossing invariant, so assume that $(alpha^j, alpha^n)$ do $epsilon$-cross by $y^(i+1)$.
    Let $y^* <= y^(i+1)$ be the $epsilon$-crossing height of $(alpha^j, alpha^n)$.

    Since the loop terminated in iteration $i$, the test at @shadow must have passed. Therefore,
    $alpha^i_+(z) <= alpha^n_+(z)$ for all $z in [y, y^(i+1)]$. Then the definition of $y^*$ ensures that

    $
    alpha^j_-(y^*) >= alpha^n_+(y^*) >= alpha^i_+(y^*),
    $

    and so $(alpha^j, alpha^i)$ $epsilon$-cross by $y^*$. Since we assumed that $(alpha^1, ..., alpha^(n-1))$ satisfy
    the crossing invariant, there is an event between $j$ and $i$ with height at most $y^*$. This event is
    also between $j$ and $n$, and so it witnesses the crossing invariant for $j$ and $n$.
]

=== Intersection scans, early intersection version

TODO: this is a re-write of an earlier section, and should get merged there
TODO: define a crossing that isn't an $epsilon$-crossing

Fix $y$ and $epsilon$, and
suppose we have a collection of lines $(alpha^i, ..., alpha^n)$ that satisfy the ordering invariant
at $y$, and suppose also that $(alpha^(i+1), ..., alpha^n)$
satisfy the crossing invariant at $y$.
To make the whole collection satisfy both invariants, we run the following algorithm.
We call this an *intersection scan to the right*.

#pseudocode-list[
  + *for* $j = i+1$ up to $n$
    + #line-label(<w-def>) let $w^j$ be the smallest height of any event between $i$ and $j$

    + *if* $(alpha^i, alpha^j_(+,epsilon))$ cross by $w^j$
      + choose $z$ before the crossing, such that $alpha^i approx_(z,epsilon) alpha^j$
      + insert an intersection event for $(alpha^i, alpha^j)$ at $z$

    + #line-label(<protect>) *if* $alpha^i (z) <= alpha^j_-(z)$ for all $z in [y, w^j]$
      + *break*
]

#inexact[
The test at @protect can be seen as an early-stopping optimization, and is not necessary for correctness.
In particular, if it is difficult to evaluate the test exactly then an approximation with no false positives
is also fine.
]

#lemma[
Suppose that $(alpha^i, ..., alpha^n)$ satisfy the ordering invariant
at $y$, and suppose also that $(alpha^(i+1), ..., alpha^n)$
satisfy the crossing invariant at $y$.
After running an intersection scan to the right,
$(alpha^i, ..., alpha^n)$ satisfy the crossing invariant at $y$.

In fact, $(alpha^i, ..., alpha^n)$ satisfy a slightly stronger crossing invariant at $y$: if for every $j > i$
if $(alpha^i, alpha^j_(+,epsilon))$ cross then the event queue contains an event between $i$ and $j$, and before
 the crossing height.
]<lem-intersection-scan2>

(The special thing about the stronger crossing invariant is that it asks whether
$(alpha^i, alpha^j_(+,epsilon))$ cross, where the usual crossing invariant asks
whether 
$(alpha^i_(-,epsilon), alpha^j_(+,epsilon))$ cross.)

#proof[
  It suffices to check the stronger crossing invariant.
  So take some $k > i$
  that $(alpha^i, alpha^k_(+,epsilon))$ cross. We consider two cases: whether or not the loop terminated
  before reaching $k$.

  - Suppose the loop terminated at some $j < k$. If $(alpha^i, alpha^k_(+,epsilon))$ cross
    after $w^j$, then the definition of $w^j$ ensures that there is an event between $i$ and $j$ (and therefore between
    $i$ and $k$) before the crossing. On the other hand, the termination condition ensures that
    $alpha^i (z) <= alpha^j_-(z)$ until $w^j$, and so if $(alpha^i, alpha^k_(+,epsilon))$ cross before $w^j$ then
    also $(alpha^j, alpha^k)$ cross before that. In this case, the crossing invariant for $(alpha^(i+1), ..., alpha^n)$
    implies the existence of an event between $j$ and $k$ (and therefore between $i$ and $k$) before the crossing.
  - If the loop included the case $j = k$, we break into two more cases:
    - If $(alpha^i, alpha^k_(+,epsilon))$ cross by $w^j$, then the algorithm inserted a witnessing event between $i$ and $j$.
    - Otherwise, the definition of $w^j$ ensures that there is an event between $i$ and $j$ before the crossing.
]
