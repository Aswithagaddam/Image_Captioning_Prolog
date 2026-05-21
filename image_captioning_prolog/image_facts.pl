% ============================================================
% Prolog facts generated from image: test_img_01
% Caption: a brown and white dog sitting on a dirt road
% ============================================================

caption('test_img_01', 'a brown and white dog sitting on a dirt road').

scene('test_img_01', outdoor).

% Detected objects (nouns from caption)
object('test_img_01', dirt).
object('test_img_01', road).
object('test_img_01', dog).

% Detected actions (verbs from caption)
action('test_img_01', sit).

% Detected attributes (adjectives from caption)
attribute('test_img_01', general, brown).
attribute('test_img_01', general, white).

% Detected relations (subject-verb-object triples)
relation('test_img_01', dog, sit, on).
