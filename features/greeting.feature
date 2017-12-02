Feature: greeting
    Recognize a face and greet the person

    Scenario: see and greet a known person (Slava)
        Given a robot
        When it sees a face of Slava
        Then it says "hi slava"
    
    Scenario: see and greet a known person (Zoe)
        Given a robot
        When it sees a face of Zoe
        Then it says "hi zoe"

    Scenario: capture photo
    	Given a robot
    	When it looks at Zoe
    	Then it says "hi zoe"
    	