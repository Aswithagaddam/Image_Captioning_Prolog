% ============================================================
% Prolog KB — epoch 1 validation captions
% ============================================================

:- dynamic object/2.
:- dynamic action/2.
:- dynamic attribute/3.
:- dynamic scene/2.
:- dynamic relation/4.
:- dynamic caption/2.

% cc_000000 — bride & bridesmaid enjoying the photo booth at a wedding reception
caption('cc_000000', 'bride & bridesmaid enjoying the photo booth at a wedding reception').
scene('cc_000000', unknown).
object('cc_000000', photo).
object('cc_000000', wedding).
object('cc_000000', bride).
object('cc_000000', booth).
object('cc_000000', reception).
action('cc_000000', enjoy).
action('cc_000000', bridesmaid).
relation('cc_000000', bride, enjoy, booth).
relation('cc_000000', bride, enjoy, at).

% cc_000001 — a city and a suburban rooftop fitted with off grid solar energy
caption('cc_000001', 'a city and a suburban rooftop fitted with off grid solar energy').
scene('cc_000001', urban).
object('cc_000001', rooftop).
object('cc_000001', grid).
object('cc_000001', energy).
object('cc_000001', city).
action('cc_000001', fit).
attribute('cc_000001', general, suburban).
attribute('cc_000001', general, solar).

% cc_000002 — welcome to vintage rusty metal sign on a white background
caption('cc_000002', 'welcome to vintage rusty metal sign on a white background').
scene('cc_000002', unknown).
object('cc_000002', background).
object('cc_000002', metal).
object('cc_000002', sign).
action('cc_000002', welcome).
attribute('cc_000002', general, rusty).
attribute('cc_000002', general, vintage).
attribute('cc_000002', general, white).

% cc_000003 — marine iguana basking in the sunshine to warm up before feeding
caption('cc_000003', 'marine iguana basking in the sunshine to warm up before feeding').
scene('cc_000003', unknown).
object('cc_000003', iguana).
object('cc_000003', sunshine).
object('cc_000003', marine).
action('cc_000003', feed).
action('cc_000003', warm).
action('cc_000003', bask).
relation('cc_000003', iguana, bask, in).

% cc_000004 — and its surrounding waters were declared an ancestral domain for ethnicity
caption('cc_000004', 'and its surrounding waters were declared an ancestral domain for ethnicity').
scene('cc_000004', urban).
object('cc_000004', waters).
object('cc_000004', ethnicity).
object('cc_000004', domain).
action('cc_000004', surround).
action('cc_000004', declare).
attribute('cc_000004', general, ancestral).

% cc_000005 — view of the village by painting artist
caption('cc_000005', 'view of the village by painting artist').
scene('cc_000005', unknown).
object('cc_000005', artist).
object('cc_000005', village).
object('cc_000005', view).
action('cc_000005', paint).

% cc_000006 — the lobby was decorated for western christian holiday .
caption('cc_000006', 'the lobby was decorated for western christian holiday .').
scene('cc_000006', unknown).
object('cc_000006', lobby).
object('cc_000006', holiday).
action('cc_000006', decorate).
attribute('cc_000006', general, christian).
attribute('cc_000006', general, western).

% cc_000007 — some of the girls at made to code
caption('cc_000007', 'some of the girls at made to code').
scene('cc_000007', unknown).
object('cc_000007', girls).
action('cc_000007', make).
action('cc_000007', code).

% cc_000008 — a beautiful night brought out the crowds as temperatures dropped into the sixties .
caption('cc_000008', 'a beautiful night brought out the crowds as temperatures dropped into the sixties .').
scene('cc_000008', unknown).
object('cc_000008', sixties).
object('cc_000008', crowds).
object('cc_000008', temperatures).
object('cc_000008', night).
action('cc_000008', drop).
action('cc_000008', bring).
attribute('cc_000008', general, beautiful).
relation('cc_000008', night, bring, crowds).
relation('cc_000008', temperatures, drop, into).

% cc_000009 — got off to a wet start
caption('cc_000009', 'got off to a wet start').
scene('cc_000009', unknown).
object('cc_000009', start).
action('cc_000009', get).
attribute('cc_000009', general, wet).

% cc_000010 — hand drawing of a raging bull
caption('cc_000010', 'hand drawing of a raging bull').
scene('cc_000010', unknown).
object('cc_000010', hand).
object('cc_000010', bull).
object('cc_000010', drawing).
action('cc_000010', rag).

% cc_000011 — illustration of an ice cream #
caption('cc_000011', 'illustration of an ice cream #').
scene('cc_000011', unknown).
object('cc_000011', cream).
object('cc_000011', illustration).
object('cc_000011', ice).

% cc_000012 — plant dense cluster hanging in the water
caption('cc_000012', 'plant dense cluster hanging in the water').
scene('cc_000012', unknown).
object('cc_000012', water).
object('cc_000012', cluster).
action('cc_000012', plant).
action('cc_000012', hang).
attribute('cc_000012', general, dense).

% cc_000013 — i could see something like this for your hair with maybe a fun headband or clip .
caption('cc_000013', 'i could see something like this for your hair with maybe a fun headband or clip .').
scene('cc_000013', unknown).
object('cc_000013', headband).
object('cc_000013', hair).
object('cc_000013', clip).
action('cc_000013', see).
attribute('cc_000013', general, fun).
relation('cc_000013', i, see, something).
relation('cc_000013', i, see, for).
relation('cc_000013', i, see, with).

% cc_000014 — this but with a little less man cave , more windows , darker walls and wood .
caption('cc_000014', 'this but with a little less man cave , more windows , darker walls and wood .').
scene('cc_000014', unknown).
object('cc_000014', windows).
object('cc_000014', walls).
object('cc_000014', man).
object('cc_000014', wood).
action('cc_000014', cave).
attribute('cc_000014', general, little).
attribute('cc_000014', general, less).
attribute('cc_000014', general, more).
attribute('cc_000014', general, darker).

% cc_000015 — aerial view of sculpture on a summer spring day at sunset .
caption('cc_000015', 'aerial view of sculpture on a summer spring day at sunset .').
scene('cc_000015', unknown).
object('cc_000015', sculpture).
object('cc_000015', summer).
object('cc_000015', day).
object('cc_000015', spring).
object('cc_000015', sunset).
object('cc_000015', view).
attribute('cc_000015', general, aerial).

% cc_000016 — happy cat in a hat holding a glass of wine and peeking from behind empty board .
caption('cc_000016', 'happy cat in a hat holding a glass of wine and peeking from behind empty board .').
scene('cc_000016', unknown).
object('cc_000016', wine).
object('cc_000016', cat).
object('cc_000016', hat).
object('cc_000016', board).
object('cc_000016', glass).
action('cc_000016', hold).
action('cc_000016', peek).
attribute('cc_000016', general, happy).
attribute('cc_000016', general, empty).

% cc_000017 — the cover of issue by person
caption('cc_000017', 'the cover of issue by person').
scene('cc_000017', unknown).
object('cc_000017', issue).
object('cc_000017', person).
object('cc_000017', cover).

% cc_000018 — welcome in from the cold .
caption('cc_000018', 'welcome in from the cold .').
scene('cc_000018', unknown).
object('cc_000018', cold).
action('cc_000018', welcome).

% cc_000019 — life 's a beach : the iconic dome against a blue sky this summer .
caption('cc_000019', 'life \'s a beach : the iconic dome against a blue sky this summer .').
scene('cc_000019', outdoor).
object('cc_000019', sky).
object('cc_000019', summer).
object('cc_000019', beach).
object('cc_000019', dome).
object('cc_000019', life).
attribute('cc_000019', general, iconic).
attribute('cc_000019', general, blue).
