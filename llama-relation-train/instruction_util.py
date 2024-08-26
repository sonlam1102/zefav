def format_few_rel(example):
    def get_head_and_tail(example):
        tokens = example['tokens']
        if len(example['head']['indices'][0]) > 1:
            head = tokens[example['head']['indices'][0][0]:example['head']['indices'][0][-1]+1]
            head = ' '.join(head)
        else:
            head = tokens[example['head']['indices'][0][0]]
            head = ''.join(head)
        if len(example['tail']['indices'][0]) > 1:
            tail = tokens[example['tail']['indices'][0][0]:example['tail']['indices'][0][-1]+1]
            tail = ' '.join(tail)
        else:
            tail = tokens[example['tail']['indices'][0][0]]
            tail = ''.join(tail)
        return head.replace(",", ""), tail.replace(",", "")

    def get_relation(example):
        return example['names'][0]

    def get_sentence(example):
        tokens = example['tokens']
        return ' '.join(tokens) + "\n"
    
    return f"""### Instruction: Given a sentence, please identify the head and tail entities in the sentence, 
    and classify the relation type into one of appropriate categories; The collection of categories is: [
    screenwriter, has part, said to be the same as, composer, participating team, headquarters location, 
    heritage designation, after a work by, participant, part of, performer, work location, operating system, 
    instance of, original language of film or TV show, follows, country of citizenship, residence, architect, 
    position held, genre, original network, main subject, sport, mountain range, publisher, manufacturer, 
    located on terrain feature, instrument, country of origin, position played on team / speciality, 
    developer, military branch, movement, distributor, owned by, platform, 
    located in or next to body of water, nominated for, location, place served by transport hub, 
    league, religion, military rank, successful candidate, operator, country, sibling, mouth of the watercourse, 
    constellation, child, notable work, field of work, subsidiary, winner, director, crosses, 
    member of political party, licensed to broadcast to, tributary, location of formation, 
    spouse, sports season of league or competition, language of work or name, occupation, 
    head of government, occupant, mother, competition class, located in the administrative territorial entity, 
    contains administrative territorial entity, participant of, voice type, followed by, member of, father, 
    record label, taxon rank, characters, applies to jurisdiction]; 
    Sentence: {get_sentence(example)}\n ### Response: ({get_head_and_tail(example)[0]}, {get_relation(example)}, {get_head_and_tail(example)[1]})"""
