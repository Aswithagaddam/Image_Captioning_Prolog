% ============================================================
% knowledge_base.pl — Image Captioning Inference Rules
% ============================================================

:- dynamic object/2.
:- dynamic action/2.
:- dynamic attribute/3.
:- dynamic scene/2.
:- dynamic relation/4.
:- dynamic caption/2.

% --- Core inference rules ---

active_image(Id) :-
    action(Id, _).

natural_scene(Id) :-
    scene(Id, outdoor) ; scene(Id, nature).

contains_animal(Id) :-
    Animals = [dog,cat,bird,horse,fish,elephant,lion,tiger,rabbit,cow,bear,deer,sheep,goat],
    member(A, Animals),
    object(Id, A).

crowded(Id) :-
    findall(O, object(Id, O), Objs),
    length(Objs, N), N > 4.

% Emotion/mood inferences from adjectives
happy_scene(Id) :-
    HappyWords = [happy,smiling,laughing,sunny,bright,colorful,playful,joyful],
    member(W, HappyWords),
    attribute(Id, _, W).

calm_scene(Id) :-
    CalmWords = [calm,quiet,peaceful,serene,still,empty,lone],
    member(W, CalmWords),
    attribute(Id, _, W).

% Image category inference
image_category(Id, animal_scene)     :- contains_animal(Id), natural_scene(Id).
image_category(Id, urban_action)     :- active_image(Id), scene(Id, urban).
image_category(Id, indoor_activity)  :- active_image(Id), scene(Id, indoor).
image_category(Id, outdoor_activity) :- active_image(Id), scene(Id, outdoor).
image_category(Id, nature_scene)     :- natural_scene(Id), \+ active_image(Id).
image_category(Id, general)          :- \+ contains_animal(Id), \+ active_image(Id).

% Complexity score: count total facts for this image
complexity_score(Id, Score) :-
    findall(O, object(Id, O), Objs),
    findall(A, action(Id, A), Acts),
    findall(R, relation(Id, R, _, _), Rels),
    length(Objs, NO), length(Acts, NA), length(Rels, NR),
    Score is NO + NA + (NR * 2).
