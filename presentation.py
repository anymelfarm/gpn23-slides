import math
from typing_extensions import Self

from manim_slides import *
from manim import *

# :)
BG = ManimColor.from_rgb((255, 255, 255))
# BG = ManimColor.from_rgb((245, 240, 245))
TEXT = ManimColor.from_rgb((0, 0, 0))
# TEXT = ManimColor.from_rgb((4, 28, 212))
TEXT_SECONDARY = ManimColor.from_rgb((250, 10, 250))

RUN_TIME = 0.25

# constants for greek letters
delta = "δ"
gamma = "γ"
epsilon = "ε"


class CircleNode(VMobject):
    def __init__(self, name=None, port_len=0.3, **kwargs):
        super().__init__(**kwargs)
        self.shape = Circle(radius=0.5, color=TEXT)
        self.add(self.shape)

        # draw principal port at top
        tip = self.shape.get_top()
        tip_port = Line(tip, tip + UP * port_len, color=TEXT)
        self.add(tip_port)
        self.port_line = tip_port

        if name is not None:
            name_text = Text(name, font_size=24, color=TEXT)
            self.add(name_text)

    def get_port_position(self):
        """
        :param port_index: the index of the port to get the position of
        :return: the position of the port
        """
        return self.port_line.get_end()


class PolarizedCircleNode(VMobject):
    def __init__(self, name=None, port_len=0.3, polarization=True, **kwargs):
        super().__init__(**kwargs)
        self.shape = Circle(radius=0.5, color=TEXT)
        self.add(self.shape)

        # draw principal port at top
        tip = self.shape.get_top()
        tip_port = Arrow(
            tip,
            tip + UP * port_len,
            color=TEXT,
            stroke_width=4,
            max_stroke_width_to_length_ratio=20,
            max_tip_length_to_length_ratio=0.4,
        )
        if not polarization:
            tip_port.rotate(PI)

        self.add(tip_port)
        self.port_line = tip_port

        if name is not None:
            name_text = Text(name, font_size=24, color=TEXT)
            self.add(name_text)

    def get_port_position(self):
        """
        :param port_index: the index of the port to get the position of
        :return: the position of the port
        """
        return self.port_line.get_end()


# JAJA ich weiß das ist ganz schrecklicher code, allerdings hatte ich ganz viele duplikatoren
class PolarizedNode(VMobject):
    def __init__(
        self,
        nports=2,
        polarities=[True, False, True],
        round_radius=0.2,
        port_buf=0.1,
        port_len=0.3,
        name=None,
        fill=None,
        **kwargs
    ):

        if len(polarities) != nports + 1:
            raise ValueError("Polarities must match the number of ports")
        super().__init__(**kwargs)
        self.nports = nports
        self.shape = RoundedTriangle(round_radius=round_radius)
        self.shape.set_fill(fill if fill is not None else BG, opacity=1.0)
        self.add(self.shape)
        self.polarities = polarities

        self.port_lines = []

        tip = self.shape.get_tip()
        tip_port = Arrow(
            tip,
            tip + UP * 0.3,
            color=TEXT,
            stroke_width=4,
            max_stroke_width_to_length_ratio=20,
            max_tip_length_to_length_ratio=0.4,
        )

        if not polarities[0]:
            tip_port.rotate(PI)

        self.add(tip_port)
        self.port_lines.insert(0, tip_port)

        self.set_n_ports(nports, polarities[1:])

        if name is not None:
            name_text = Text(name, font_size=24, color=TEXT)
            self.add(name_text)

        self.set_stroke(color=TEXT)

    def set_n_ports(self, nports, polarities):
        """
        Sets the number of ports of the node.
        :param nports: the number of ports to set
        """

        self.nports = nports + 1

        # remove old port lines
        for port_line in self.port_lines[1:]:
            self.remove(port_line)

        # add new port lines
        self.port_lines = [self.port_lines[0]]
        l0 = self.shape.get_port_line()[0]
        r0 = self.shape.get_port_line()[1]

        l = l0 + (r0 - l0) * 0.1
        r = r0 - (r0 - l0) * 0.1

        distance = (r - l) / (self.nports - 2) if self.nports > 1 else 0

        for i in range(self.nports - 1):
            port = l + i * distance
            port_line = Arrow(
                port,
                port + DOWN * 0.3,
                color=TEXT,
                stroke_width=4,
                max_stroke_width_to_length_ratio=20,
                max_tip_length_to_length_ratio=0.4,
            )
            if polarities[i]:
                port_line.rotate(PI)
            self.add(port_line)
            self.port_lines.append(port_line)

    def add_name(self, name):
        """
        Adds a name to the node.
        :param name: the name to add
        """
        name_text = Text(name, font_size=24, color=TEXT)
        tip = self.shape.get_tip()
        name_text.move_to(tip + UP * 0.3 + RIGHT * 0.5)
        self.add(name_text)

    def get_port_position(self, port_index):
        """
        :param port_index: the index of the port to get the position of
        :return: the position of the port
        """
        if port_index > self.nports:
            raise ValueError("Port index out of range")
        pol = self.polarities[port_index]
        if port_index == 0:
            pol = not pol
        if pol:
            return self.port_lines[port_index].get_start()
        return self.port_lines[port_index].get_end()


# a class that combines a rounded triangle with ports
class Node(VMobject):
    def __init__(
        self,
        nports=2,
        round_radius=0.2,
        port_buf=0.1,
        port_len=0.3,
        name=None,
        fill=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nports = nports
        self.shape = RoundedTriangle(round_radius=round_radius)
        self.shape.set_fill(fill if fill is not None else BG, opacity=1.0)
        self.add(self.shape)

        self.port_lines = []

        tip = self.shape.get_tip()
        tip_port = Line(tip, tip + UP * 0.3)
        self.add(tip_port)
        self.port_lines.insert(0, tip_port)

        self.set_n_ports(nports)

        if name is not None:
            name_text = Text(name, font_size=24, color=TEXT)
            self.add(name_text)

        self.set_stroke(color=TEXT)

    def set_n_ports(self, nports):
        """
        Sets the number of ports of the node.
        :param nports: the number of ports to set
        """

        self.nports = nports + 1

        # remove old port lines
        for port_line in self.port_lines[1:]:
            self.remove(port_line)

        # add new port lines
        self.port_lines = [self.port_lines[0]]
        l0 = self.shape.get_port_line()[0]
        r0 = self.shape.get_port_line()[1]

        l = l0 + (r0 - l0) * 0.1
        r = r0 - (r0 - l0) * 0.1

        distance = (r - l) / (self.nports - 2) if self.nports > 1 else 0

        for i in range(self.nports - 1):
            port = l + i * distance
            port_line = Line(port, port + DOWN * 0.3, color=TEXT)
            self.add(port_line)
            self.port_lines.append(port_line)

    def add_name(self, name):
        """
        Adds a name to the node.
        :param name: the name to add
        """
        name_text = Text(name, font_size=24, color=TEXT)
        tip = self.shape.get_tip()
        name_text.move_to(tip + UP * 0.3 + RIGHT * 0.5)
        self.add(name_text)

    def get_port_position(self, port_index):
        """
        :param port_index: the index of the port to get the position of
        :return: the position of the port
        """
        if port_index > self.nports:
            raise ValueError("Port index out of range")
        return self.port_lines[port_index].get_end()

    def hide_ports(self):
        """
        Hides the port lines of the triangle by setting the end of the port lines to the start point.
        """
        for port_line in self.port_lines:
            x = port_line.get_start()
            port_line.put_start_and_end_on(x, x)


class RoundedTriangle(ArcPolygon):
    def __init__(
        self,
        p1=LEFT * math.sin(math.radians(60))
        + DOWN * math.cos(math.radians(60)),
        p2=RIGHT * math.sin(math.radians(60))
        + DOWN * math.cos(math.radians(60)),
        p3=UP,
        round_radius=0.1,
        **kwargs
    ):
        a = (p2 - p1) * round_radius + p1
        b = (p3 - p1) * round_radius + p1
        c = (p3 - p2) * round_radius + p2
        d = (p1 - p2) * round_radius + p2

        self.port_line_a = a
        self.port_line_d = d
        self.tip = p3

        super().__init__(
            b,
            p3,
            c,
            d,
            a,
            arc_config=[
                {"angle": 0 * DEGREES},
                {"angle": 0 * DEGREES},
                {"angle": -90 * DEGREES},
                {"angle": 0 * DEGREES},
                {"angle": -90 * DEGREES},
            ],
            **kwargs,
        )

    def get_port_line(self):
        return self.port_line_a, self.port_line_d

    def get_tip(self):
        return self.tip


def connect_nodes(node1: Node, node2: Node, port1: int = 0, port2: int = 0):
    """
    Connects two nodes at the specified ports.
    :param node1: the first node
    :param node2: the second node
    :param port1: the port index of the first node
    :param port2: the port index of the second node
    """
    if port1 >= node1.nports or port2 >= node2.nports:
        raise ValueError("Port index out of range")

    if port1 == 0 and port2 == 0:
        # connect the principal ports
        node1.rotate(PI)
        p1 = node1.get_port_position(port1)
        p2 = node2.get_port_position(port2)
        dist = p2 - p1
        node2.move_to(node2.get_center() - dist)
    else:
        raise NotImplementedError(
            "Only connecting principal ports is implemented for now."
        )

    g = VGroup(node1, node2)
    g.move_to(ORIGIN)

    return g


def duplicate_animation(gamma: Node, delta: Node, scene):
    delta2 = delta.copy()
    delta2.next_to(delta, RIGHT, buff=0.5)
    gamma2 = gamma.copy()
    gamma2.next_to(gamma, RIGHT, buff=0.5)

    nodes = VGroup(gamma, delta, delta2, gamma2)

    scene.add(gamma2, delta2)

    # move gamma to delta2 and delta2 to gamma
    prev_center = gamma.get_center()
    gamma.generate_target()
    gamma.target.move_to(delta2.get_center())
    delta2.generate_target()
    delta2.target.move_to(prev_center)

    # move gamma2 to delta and delta to gamma2
    prev_center = gamma2.get_center()
    gamma2.generate_target()
    gamma2.target.move_to(delta.get_center())
    delta.generate_target()
    delta.target.move_to(prev_center)

    for node in nodes:
        node.hide_ports()
        for port_line in node.port_lines[1:]:
            port_line.set_stroke(color=BG, width=0)
            scene.remove(port_line)

    FadeOut(gamma2)
    FadeOut(delta2)

    scene.play(
        FadeIn(gamma2),
        FadeIn(delta2),
        MoveToTarget(gamma),
        MoveToTarget(delta),
        MoveToTarget(gamma2),
        MoveToTarget(delta2),
    )

    return nodes


class Main(Slide):
    def show_code_steps(
        self, codes, lang="JavaScript", align=lambda x: x.to_edge(UP)
    ):
        code_boxes = []

        for code_str in codes:
            code = Code(
                code_string=code_str,
                tab_width=4,
                language=lang,
                add_line_numbers=False,
            )
            box = VGroup(code).scale(1.3)
            code_boxes.append(box)

        align(code_boxes[0])
        self.play(FadeIn(code_boxes[0]))
        self.next_slide()

        for i in range(1, len(code_boxes)):
            code_boxes[i].next_to(code_boxes[i - 1], DOWN, buff=0.4)

            self.play(FadeIn(code_boxes[i]))
            self.next_slide()
            self.wait()
        return VGroup(*code_boxes)

    def construct(self):
        # initialization
        self.camera.background_color = BG
        self.wait_time_between_slides = 0.01
        Text.set_default(color=TEXT, font_size=40, font="Linux Libertine")
        MathTex.set_default(color=TEXT, font_size=40)

        Write.set_default(run_time=RUN_TIME)
        FadeIn.set_default(run_time=RUN_TIME)
        FadeOut.set_default(run_time=RUN_TIME)
        Animation.set_default(run_time=RUN_TIME * 2)  # TODO: not sure abou this
        #
        # title = Text("Interaction Combinators", font_size=60)
        # title.set_color(TEXT)

        # subtitle = Text("The Hidden Patterns of Computation?", font_size=40)
        # subtitle.set_color(text)
        # subtitle.next_to(title, DOWN, buff=0.5)

        # group the title and subtitle
        # title_group = VGroup(title, subtitle)
        # title_group.arrange(DOWN, buff=0.5)

        # self.add(title_group)
        # self.wait(1)

        # =============
        self.next_slide()
        # =============

        # self.play(FadeOut(title), FadeOut(subtitle))

        computation = Text("What is Computation?")

        self.play(FadeIn(computation))

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(computation))

        # TODO: use MathTex
        simple_equation = Text("2 + 4 = ?", font_size=60)

        four = simple_equation[2]
        four.set_color(TEXT_SECONDARY)
        start = simple_equation[0:2]
        end = simple_equation[3:]

        # TODO: use MathTex
        four_repr = Text("1 + 1 + 1 + 1", font_size=60, color=TEXT_SECONDARY)

        self.play(Write(simple_equation))

        # ============
        self.next_slide()
        # ============

        padding = 1.4 * (start.width / 2)
        self.play(
            # muss eigentlich iwie besser gehen
            start.animate.move_to(
                four_repr.get_edge_center(LEFT)
                + LEFT * (padding + start.width / 2)
            ),
            end.animate.move_to(
                four_repr.get_edge_center(RIGHT)
                + RIGHT * (padding + end.width / 2)
            ),
        )

        self.play(Transform(four, four_repr), run_time=0.5)

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(simple_equation), FadeOut(four_repr))
        # title_peano = Text("The Peano Axiom", font_size=40)
        # subtitle_peano = Text("What even are numbers?", font_size=30)
        # title_peano.set_color(text)
        # subtitle_peano.set_color(text)
        # # place the title at the top of the screen
        # title_peano.to_edge(UP)
        # subtitle_peano.next_to(title_peano, DOWN, buff=0.3)
        # # move to the left side of the screen
        # title_peano.to_edge(LEFT)
        # subtitle_peano.to_edge(LEFT)

        # self.play(
        #     FadeIn(title_peano), FadeIn(subtitle_peano)
        # )

        peano_group = VGroup(
            MathTex(r"0 = \text{Zero}"),
            MathTex(r"1 = \text{Successor of Zero}"),
            MathTex(r"2 = \text{Successor of Successor of Zero}"),
        )

        peano_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        # ============
        self.next_slide()
        # ============

        self.play(LaggedStartMap(FadeIn, peano_group, lag_ratio=0.1))

        peano_group_replacement = VGroup(
            MathTex("0 = Z"),
            MathTex("1 = S(Z)"),
            MathTex("2 = S(S(Z))"),
        )

        peano_group_replacement.arrange(DOWN, aligned_edge=LEFT, buff=0.5)

        # ============
        self.next_slide()
        # ============

        self.play(Transform(peano_group, peano_group_replacement, run_time=0.5))

        # ============
        self.next_slide()
        # ============

        # subtitle_peano_add = Text("Let's define addition", font_size=30)
        # subtitle_peano_add.next_to(title_peano, DOWN, buff=0.3)
        # subtitle_peano_add.to_edge(LEFT)
        # subtitle_peano_add.set_color(text)

        self.play(
            FadeOut(peano_group_replacement),
            FadeOut(peano_group),
            # Transform(subtitle_peano, subtitle_peano_add, run_time=2),
        )

        addition_def = VGroup(
            MathTex(r"\textsf{add}(k, 0) = k"),
            MathTex(r"\textsf{add}(k, S(n)) = S(\textsf{add}(k, n))"),
        )

        addition_def.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(LaggedStartMap(FadeIn, addition_def, lag_ratio=0.1))

        # ============
        self.next_slide()
        # ============

        self.play(
            FadeOut(addition_def),
        )

        peano_calculation = MathTex(
            r"2 + 4 = \textsf{add}(S(S(Z)), S(S(S(S(Z)))) =\ ?, ", font_size=50
        ).to_edge(UP, buff=0.5)
        peano_calculation_solved = VGroup(
            MathTex(r"= S(\textsf{add}(S(S(Z)), S(S(S(Z)))), "),
            MathTex(r"= S(S(\textsf{add}(S(S(Z)), S(S(Z)))), "),
            MathTex(r"= S(S(S(\textsf{add}(S(S(Z)), S(Z)))), "),
            MathTex(r"= S(S(S(S(\textsf{add}(S(S(Z)), Z))))), "),
            MathTex(r"= S(S(S(S(S(S(Z))))))"),
        )

        peano_calculation_solved.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        peano_calculation_solved.next_to(peano_calculation, DOWN, buff=0.5)

        self.play(Write(peano_calculation))

        # ============
        self.next_slide()
        # ============

        self.play(
            LaggedStartMap(FadeIn, peano_calculation_solved, lag_ratio=0.1)
        )

        # ============
        self.next_slide()
        # ============

        self.play(
            FadeOut(peano_calculation),
            FadeOut(peano_calculation_solved),
        )

        # TODO: where last arrow, it worked before!!
        code_steps = self.show_code_steps(
            codes=[
                "function add(x, y) { return x + y }\nconsole.log(add(2, 4))",
                "const add = (x, y) => x + y\nconsole.log(add(2, 4))",
                "const add = x => y => x + y\nconsole.log(add(2)(4))",
            ]
        )

        # ============
        self.next_slide()
        # ============

        # example beta reduction
        exA = MathTex(r"(x\Rightarrow y\Rightarrow x)(42)")
        exB = MathTex(r"\leadsto(y\Rightarrow 42)")
        exB.next_to(exA, RIGHT)

        self.play(FadeOut(code_steps), Write(exA))
        self.next_slide()
        self.play(Write(exB))

        ex = VGroup(exA, exB)
        self.play(ex.animate.move_to(ORIGIN))

        # in general:
        self.next_slide()
        ruleA = MathTex(r"(x\Rightarrow M)(N)")
        ruleA.next_to(exA, DOWN, buff=0.5)
        ruleB = MathTex(r"\leadsto M[x\mapsto N]")
        ruleB.next_to(ruleA, RIGHT)

        self.play(Write(ruleA))
        # self.next_slide()
        self.play(Write(ruleB))

        rule = VGroup(ruleA, ruleB)

        # focus on beta reduction
        self.next_slide()
        self.play(FadeOut(ex), rule.animate.move_to(ORIGIN).scale(2))

        # ============
        self.next_slide()
        # ============

        self.play(rule.animate.scale(0.75).to_corner(UL))

        # TODO: better alignment

        subText = Text("Substitution").next_to(
            rule, DOWN, aligned_edge=LEFT, buff=0.5
        )
        subDesc = MathTex(
            r"(x\Rightarrow y\Rightarrow x)(42)\leadsto(y\Rightarrow 42)"
        ).next_to(subText, RIGHT, buff=0.5)

        eraText = Text("Erasure").next_to(
            subText, DOWN, aligned_edge=LEFT, buff=0.5
        )
        eraDesc = MathTex(
            r"(x\Rightarrow y\Rightarrow y)(42)\leadsto(y\Rightarrow y)"
        ).next_to(subDesc, DOWN, aligned_edge=LEFT, buff=0.5)

        dupText = Text("Duplication").next_to(
            eraText, DOWN, aligned_edge=LEFT, buff=0.5
        )
        dupDesc = MathTex(
            r"(x\Rightarrow\textsf{add}(x)(x))(42)\leadsto\textsf{add}(42)(42)"
        ).next_to(eraDesc, DOWN, aligned_edge=LEFT, buff=0.5)

        shaText = Text("Sharing").next_to(
            dupText, DOWN, aligned_edge=LEFT, buff=0.5
        )
        # you don't want to duplicate it, but also do not evaluate it if it is not used!
        shaDesc = MathTex(
            r"(x\Rightarrow \textsf{add}(x)(x))(\textsf{fac}(42))"
        ).next_to(dupDesc, DOWN, aligned_edge=LEFT, buff=0.5)

        ordText = Text("Order").next_to(
            shaText, DOWN, aligned_edge=LEFT, buff=0.5
        )

        combined = VGroup(
            subText,
            subDesc,
            dupText,
            dupDesc,
            eraText,
            eraDesc,
            shaText,
            shaDesc,
            ordText,
        )

        self.play(Write(subText))
        self.next_slide()
        self.play(Write(subDesc))
        self.next_slide()
        self.play(Write(eraText))
        self.next_slide()
        self.play(Write(eraDesc))
        self.next_slide()
        self.play(Write(dupText))
        self.next_slide()
        self.play(Write(dupDesc))
        self.next_slide()

        # implicit rules!
        # - left term in application becomes abstraction -> substitution
        # - term is not bound in substitution -> erasing
        # - etc.
        # self.wait(.5)
        # self.play(Write(ordDesc))
        self.play(Write(shaText))
        self.next_slide()
        self.play(Write(shaDesc))
        self.next_slide()
        self.play(Write(ordText))
        # self.wait(.5)
        # self.play(Write(shaDesc))

        # ============
        self.next_slide()
        # ============

        self.play(
            FadeOut(rule),
            FadeOut(combined),
        )

        turing = MathTex(
            r"\text{anonymous functions}+\text{function calls}=\text{Turing Complete}!"
        ).scale(1.2)
        self.play(Write(turing))
        self.next_slide()
        self.play(turing.animate.scale(0.85).to_corner(UL))

        what_number = MathTex(
            r"\text{Example: number} = \text{duplicate argument } n\text{ times}",
        ).next_to(turing, DOWN, buff=0.5, aligned_edge=LEFT)
        # and then apply the arguments to eachother

        # - numbers increment things - make the term larger
        # - the only way to make terms larger is by duplication
        # -> represent numbers as n-times duplication: 3=(s -> z -> s(s(s(z))))

        church_three = MathTex(
            r"3=(s\Rightarrow z\Rightarrow s(s(s(z))))"
        ).move_to(ORIGIN)
        succ = MathTex(
            r"\text{incr}=(n\Rightarrow s\Rightarrow z\Rightarrow s(n(s)(z)))",
            substrings_to_isolate="s",
        ).next_to(church_three, DOWN, buff=0.5)
        succ.set_color_by_tex("s", TEXT_SECONDARY)

        self.play(Write(what_number))
        self.next_slide()
        self.play(Write(church_three))
        self.next_slide()
        self.play(Write(succ))

        # basically Peano!

        exampleA = VGroup(what_number, church_three, succ)

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(exampleA))

        what_recursion = MathTex(
            r"\text{Example: recursion} = \text{self-duplication} + \text{self-erasure} + \text{self-application}",
        ).next_to(turing, DOWN, buff=0.5, aligned_edge=LEFT)
        # erasing self is required for ending the loop!

        # of course, all the operators are implemented in anonymous functions as well
        self.play(Write(what_recursion))
        self.next_slide()
        code_steps = self.show_code_steps(
            codes=[
                "const fac = n => n < 2 ? 1 : n * fac(n - 1)",
                "const _fac = fac => n => n < 2 ? 1 : n * fac(fac)(n - 1)",
                "const fac = _fac(_fac)",  # // (f => f(f))(_fac)",
            ],
            align=lambda x: x.next_to(what_recursion, DOWN).move_to(
                [0, what_recursion.get_y() - 1.5, 0]
            ),
        )

        exampleB = VGroup(what_recursion, code_steps)

        # won't work in actual JS :)

        # TODO: what about side effects! (hint to later)

        # ============
        self.next_slide()
        # ============

        lambdaturing = MathTex(
            r"\lambda-\text{calculus}=\text{Turing Complete}!"
        ).to_corner(UL)
        self.play(Transform(turing, lambdaturing))

        # ============
        self.next_slide()
        # ============
        self.remove(turing)
        self.play(FadeOut(lambdaturing), FadeOut(exampleB))
        what_order = Text("What about reduction order?").scale(1.2)
        self.play(Write(what_order))
        self.next_slide()
        self.play(what_order.animate.scale(0.85).to_corner(UL))

        code = Code(
            code_string="(f => f(f))(fac => n => n < 2 ? 1 : n * fac(fac)(n - 1))(42)",
            tab_width=4,
            language="JavaScript",
            add_line_numbers=False,
        ).move_to([0, what_order.get_y() - 1, 0])

        inf_path1 = (
            MathTex(
                r"42\cdot\textsf{fac}(\textsf{fac})(41)\leadsto42\cdot(n\Rightarrow\dots\textsf{fac}(\textsf{fac})(\dots))(41)"
            )
            .next_to(code, DOWN)
            .shift([0.5, 0, 0])
        )
        inf_path2 = (
            MathTex(
                r"\leadsto42\cdot(n\Rightarrow\dots(n\Rightarrow\dots\textsf{fac}(\textsf{fac})(\dots))(\dots))(41)"
            )
            .next_to(inf_path1, DOWN, aligned_edge=LEFT)
            .shift([-0.5, 0, 0])
        )
        inf_path3 = MathTex(r"\leadsto\dots").next_to(
            inf_path2, DOWN, aligned_edge=LEFT
        )
        inf_path4 = MathTex(
            r"\leadsto42\cdot41\cdot40\cdot39\cdot\ldots\cdot1"
        ).next_to(inf_path3, DOWN, aligned_edge=LEFT)
        inf_result = MathTex(
            r"\leadsto1405006117752879898543142606244511569936384000000000"
        ).next_to(inf_path4, DOWN, aligned_edge=LEFT)
        self.play(Write(code))
        self.next_slide()
        self.play(Write(inf_path1))
        self.next_slide()
        self.play(Write(inf_path2))
        self.next_slide()
        # could be done infinitely times!
        self.play(Write(inf_path3))
        self.next_slide()
        # still eventually refocussable to the outer terms, yielding a result!
        self.play(Write(inf_path4))
        self.next_slide()
        self.play(Write(inf_result))
        # demo write in NodeJS terminal "(f => f(f))(fac => n => n < 2 ? 1 : n * fac(fac)(n - 1))(42)"!

        inf_paths = VGroup(
            inf_path1, inf_path2, inf_path3, inf_path4, inf_result
        )

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(inf_paths))

        red = MathTex(r"M\leadsto\dots\leadsto M'").next_to(
            code, DOWN, buff=0.2, aligned_edge=LEFT
        )

        spacing = 1.31415926

        M = MathTex("M")
        M.move_to(UP * spacing)

        M1 = MathTex("M_1")
        M1.move_to(LEFT * spacing)

        Mn = MathTex("M_n")
        Mn.move_to(LEFT * spacing + DOWN * spacing)

        M2 = MathTex("M_2")
        M2.move_to(RIGHT * spacing)

        Mm = MathTex("M_m")
        Mm.move_to(RIGHT * spacing + DOWN * spacing)

        M_PRIME = MathTex("M'")
        M_PRIME.move_to(DOWN * spacing * 2)

        # draw arrows from AC to BC and CC and from BC and CC to M_PRIME
        arrow1 = Arrow(
            M.get_bottom(),
            M1.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow2 = Arrow(
            M.get_bottom(),
            M2.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow3 = DashedLine(
            M1.get_bottom(),
            Mn.get_top(),
            color=TEXT,
            buff=0.1,
        )
        arrow4 = Arrow(
            Mn.get_bottom(),
            M_PRIME.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow5 = DashedLine(
            M2.get_bottom(),
            Mm.get_top(),
            color=TEXT,
            buff=0.1,
        )
        arrow6 = Arrow(
            Mm.get_bottom(),
            M_PRIME.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )

        # make the arrows thiner
        arrow1.set_stroke(width=2)
        arrow2.set_stroke(width=2)
        arrow3.set_stroke(width=2)
        arrow4.set_stroke(width=2)
        arrow5.set_stroke(width=2)
        arrow6.set_stroke(width=2)

        g = (
            VGroup(
                M,
                M1,
                Mn,
                M2,
                Mm,
                M_PRIME,
                arrow1,
                arrow2,
                arrow3,
                arrow4,
                arrow5,
                arrow6,
            )
            .move_to(ORIGIN)
            .shift([0, -1.5, 0])
        )

        weak = MathTex(r"n\le m!!").next_to(g, RIGHT, buff=0.5)
        opti = Text(r"optimal path?").next_to(weak, DOWN, buff=0.5)
        self.play(Write(red))
        self.next_slide()
        self.play(Create(g))
        self.wait()
        self.next_slide()
        self.play(g.animate.shift([-1.5, 0, 0]), Write(weak))
        self.next_slide()
        self.play(Write(opti))

        # BAD FOR PARALLELISM!

        # ============
        self.next_slide()
        # ============

        self.play(
            FadeOut(code),
            FadeOut(red),
            FadeOut(g),
            FadeOut(weak),
            FadeOut(opti),
        )

        title = Text("Recap / Goal").to_corner(UL)
        self.play(Transform(what_order, title))

        recap = BulletedList(
            r"$\text{Functions \& applications}\Rightarrow\text{Minimal rules}$",
            r"$\text{Beta-reduction}\Rightarrow\text{No implicit behavior}$",
            r"$\text{Implicit duplication/erasure}\Rightarrow\text{Explicit control!}$",
            r"$\text{Sharing}\Rightarrow\text{No redundancy}$",
            r"$\text{Confluence}\Rightarrow\text{One-step, parallelism, determinism}$",
        )

        self.play(Write(recap))

        # ---
        # what model of computation could be better than lambda calculus?
        # - stay as minimal as lambda calculus (only a few constructs to remember)
        # - make implicit rules by beta-reduction explicit
        # - do not support divergence for massive parallelism

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(what_order), FadeOut(recap))

        triangle: Node = Node(round_radius=0.20)
        triangle.set_stroke(TEXT)
        triangle.move_to(ORIGIN)
        triangle.scale(2)
        principal_pos = triangle.get_port_position(0)
        pp = Text("Principal Port", font_size=24, color=TEXT_SECONDARY)
        pp.next_to(principal_pos, UP, buff=0.1)

        l = triangle.get_port_position(1)
        r = triangle.get_port_position(2)

        brace = BraceBetweenPoints(l, r)
        brace.set_color(TEXT_SECONDARY)
        btip: Text = brace.get_tip()

        ap = Text("Auxiliary Ports", font_size=24, color=TEXT_SECONDARY)
        ap.next_to(btip, DOWN, buff=0.1)

        self.play(Create(triangle))
        self.play(FadeIn(pp))
        self.play(FadeIn(brace), FadeIn(ap))

        # ============
        self.next_slide()
        # ============

        self.remove(pp, ap, brace, triangle)

        triangle2 = Node(4)
        triangle2.set_stroke(TEXT)
        triangle2.scale(2)
        triangle2.move_to(ORIGIN)

        self.add(triangle2)
        self.wait()

        # ============
        self.next_slide()
        # ============

        self.remove(triangle2)

        triangle4 = Node(2, fill=TEXT)
        triangle4.scale(2)
        triangle4.move_to(ORIGIN)

        triangle5 = Node(2, fill=TEXT)
        triangle5.scale(2)
        triangle5.move_to(ORIGIN)

        group1 = VGroup(triangle4, triangle5)
        self.add(group1)
        self.wait()

        # ============
        self.next_slide()
        # ============
        self.remove(group1)

        a1 = Node(2)
        a2 = Node(2)
        a2.scale(0.5)
        a3 = CircleNode()
        a3.scale(0.5)
        a4 = Node(2, fill=TEXT)
        a4.rotate(PI)
        a4.move_to(UP * 1)

        a5 = Node(2)
        a5.scale(0.5)
        a5.rotate(PI)

        a5.move_to(a4.get_port_position(1) - a5.get_port_position(0))

        a1.move_to(DOWN * 1)

        a2.move_to(a1.get_port_position(1) - a2.get_port_position(0) + UP * 0.5)
        a3.move_to(a1.get_port_position(2) - a3.get_port_position() + UP * 0.3)

        self.add(a1, a2, a3, a4, a5)
        self.wait()

        # ============
        self.next_slide()
        # ============
        package = Square(side_length=6)
        package.set_stroke(TEXT)
        self.play(Create(package))

        p1 = a2.get_port_position(1)
        p2 = a2.get_port_position(2)
        p3 = a4.get_port_position(2)
        p4 = a5.get_port_position(1)
        p5 = a5.get_port_position(2)
        pn = []

        # from all pN draw a line to (x, x+ 3*Down)
        for p in [p1, p2]:
            dest = np.array([p[0], (3.3 * DOWN)[1], 0])
            line = Line(p, dest, color=TEXT)
            pn.append(line)

        for p in [p3, p4, p5]:
            dest = np.array([p[0], (3.3 * UP)[1], 0])
            line = Line(p, dest, color=TEXT)
            pn.append(line)

        self.play(*[Create(line) for line in pn])

        # ============
        self.next_slide()
        # ============
        package_text = Text("Package", font_size=60, color=TEXT)
        package.z_index = 10
        self.play(
            package.animate.set_fill(BG, opacity=1.0),
        )

        # ============
        self.next_slide()
        # ============
        self.remove(package, *pn, a1, a2, a3, a4, a5)

        node1 = Node(nports=2, round_radius=0.20, fill=BG)
        node2 = Node(nports=2, round_radius=0.20, fill=BG)

        connected = connect_nodes(node1, node2, port1=0, port2=0)
        self.add(connected)

        self.wait()

        # ============
        self.next_slide()
        # ============
        # draw a box around connected
        box = Rectangle(width=1.5, height=2, color=TEXT, stroke_width=4)
        # extend all portlines a oputwards
        node1_1 = node1.get_port_position(1)  # top right
        node1_2 = node1.get_port_position(2)  # top left
        node2_1 = node2.get_port_position(1)  # bottom left
        node2_2 = node2.get_port_position(2)  # bottom right

        node1_1l = Line(node1_1 + UP * -1.1, node1_1 + UP * 0.0, color=TEXT)
        node1_2l = Line(node1_2 + UP * -1.1, node1_2 + UP * 0.0, color=TEXT)
        node2_1l = Line(node2_1 + DOWN * -1.1, node2_1 + DOWN * 0.0, color=TEXT)
        node2_2l = Line(node2_2 + DOWN * -1.1, node2_2 + DOWN * 0.0, color=TEXT)

        self.play(
            FadeOut(node1, shift=DOWN, run_time=0.5),
            FadeOut(node2, shift=UP, run_time=0.5),
            FadeIn(box),
            # Fade in the lines
            FadeIn(node1_1l),
            FadeIn(node1_2l),
            FadeIn(node2_1l),
            FadeIn(node2_2l),
        )

        # ============
        self.next_slide()
        # ============
        self.remove(box, node1_1l, node1_2l, node2_1l, node2_2l)
        inter_combs_text = Text(
            "Interaction Combinators", font_size=60, color=TEXT
        )

        self.play(Write(inter_combs_text))

        # ===========
        self.next_slide()
        # ===========
        self.remove(inter_combs_text)

        duplicator = Node(nports=2, round_radius=0.20, fill=TEXT)
        constructor = Node(nports=2, round_radius=0.20)
        eraser = CircleNode()

        # the names as text
        duplicator_name = Text("Duplicator", font_size=32, color=TEXT_SECONDARY)
        constructor_name = Text(
            "Constructor", font_size=32, color=TEXT_SECONDARY
        )
        eraser_name = Text("Eraser", font_size=32, color=TEXT_SECONDARY)

        # group so that the text is in one line and lines up with the nodes

        node_group = VGroup(duplicator, constructor, eraser)
        node_group.arrange(RIGHT, buff=2.0)
        down_tt = 2 * DOWN
        duplicator_name.move_to(duplicator.get_center() + down_tt)
        constructor_name.move_to(constructor.get_center() + down_tt)
        eraser_name.move_to(eraser.get_center() + down_tt)

        self.play(
            Create(duplicator),
            Create(constructor),
            Create(eraser),
            Write(duplicator_name),
            Write(constructor_name),
            Write(eraser_name),
        )

        # ===========
        self.next_slide()
        # ============

        self.remove(
            duplicator,
            constructor,
            eraser,
            duplicator_name,
            constructor_name,
            eraser_name,
        )
        annihilation_text = Text("Annihilation", font_size=60, color=TEXT)
        self.add(annihilation_text)
        self.wait()

        # ===========
        self.next_slide()
        # ===========
        self.play(annihilation_text.animate.scale(0.75).to_corner(UL))

        node1 = Node(nports=2, round_radius=0.20, fill=GRAY)
        node2 = Node(nports=2, round_radius=0.20, fill=GRAY)
        node2.rotate(PI)
        node2.move_to(UP * 1)
        node1.move_to(DOWN * 1)

        self.add(node1, node2)

        p1 = node1.get_port_position(1)  # bottom left
        p2 = node1.get_port_position(2)  # bottom right
        p3 = node2.get_port_position(1)  # top right
        p4 = node2.get_port_position(2)  # top left

        # write x1 and x2 at p1 and p2
        x1 = MathTex("x_1", font_size=24, color=TEXT_SECONDARY)
        x2 = MathTex("x_2", font_size=24, color=TEXT_SECONDARY)
        x1.move_to(p1 + DOWN * 0.2)
        x2.move_to(p2 + DOWN * 0.2)

        x3 = MathTex("y_1", font_size=24, color=TEXT_SECONDARY)
        x4 = MathTex("y_2", font_size=24, color=TEXT_SECONDARY)
        x3.move_to(p3 + UP * 0.2)
        x4.move_to(p4 + UP * 0.2)

        self.add(x1, x2, x3, x4)

        self.wait()

        # ===========
        self.next_slide()
        # ===========

        # Calculate center (optional for aesthetics)
        center = (p1 + p2 + p3 + p4) / 4
        tightness = 1

        # Control points to curve and cross the splines
        ctrl1a = p1 + UP * tightness + RIGHT * tightness  # pull toward crossing
        ctrl1b = (
            p3 + DOWN * tightness + LEFT * tightness
        )  # pull back toward target

        ctrl2a = p2 + UP * tightness + LEFT * tightness  # opposite side curve
        ctrl2b = p4 + DOWN * tightness + RIGHT * tightness

        # Create Bezier curves that cross
        curve1 = CubicBezier(p1, ctrl1b, ctrl1a, p3).set_stroke(BLUE, 4)
        curve2 = CubicBezier(p2, ctrl2b, ctrl2a, p4).set_stroke(RED, 4)
        spline = VGroup(curve1, curve2)

        spline.set_stroke(TEXT)

        self.play(
            FadeOut(node1, shift=UP, run_time=0.5),
            FadeOut(node2, shift=DOWN, run_time=0.5),
            FadeIn(curve1, run_time=1),
            FadeIn(curve2, run_time=1),
        )

        # ===========
        self.next_slide()
        # ===========
        self.remove(x1, x2, x3, x4, curve1, curve2, node1, node2)
        eraser1 = CircleNode()
        eraser2 = CircleNode()

        eraser1.move_to(DOWN / 1.6)
        eraser2.move_to(UP / 1.6)
        eraser2.rotate(PI)

        self.add(eraser1, eraser2)

        self.wait()
        # ===========
        self.next_slide()
        # ===========

        self.play(
            FadeOut(eraser1, shift=UP, run_time=0.5),
            FadeOut(eraser2, shift=DOWN, run_time=0.5),
        )

        # ===========
        self.next_slide()
        # ===========

        self.play(FadeOut(annihilation_text))
        comm_text = Text("Commutation", font_size=60, color=TEXT)
        self.add(comm_text)
        self.wait()

        # ===========
        self.next_slide()
        # ===========
        self.play(comm_text.animate.scale(0.75).to_corner(UL))

        comb_duplication = Node(nports=2, round_radius=0.20, fill=TEXT)

        comb_constructor = Node(nports=2, round_radius=0.20)

        dup_nodes = connect_nodes(comb_constructor, comb_duplication, 0, 0)
        dup_nodes.move_to(ORIGIN)
        self.add(dup_nodes)
        self.wait()

        # ===========
        self.next_slide()
        # ===========

        comb_duplication2 = comb_duplication.copy()
        comb_constructor2 = comb_constructor.copy()

        dist = 1.5

        comb_duplication.generate_target().move_to(UP * dist + RIGHT)
        comb_duplication2.generate_target().move_to(UP * dist + LEFT)
        comb_constructor.generate_target().move_to(DOWN * dist + LEFT)
        comb_constructor2.generate_target().move_to(DOWN * dist + RIGHT)
        self.play(
            MoveToTarget(comb_constructor),
            MoveToTarget(comb_duplication),
            MoveToTarget(comb_constructor2),
            MoveToTarget(comb_duplication2),
        )

        # connect dup2 port 1 with const1 port 2 with a line
        dup2_port1 = comb_duplication2.get_port_position(1)
        const1_port2 = comb_constructor.get_port_position(2)
        line1 = Line(dup2_port1, const1_port2, color=TEXT)

        # connect dup1 port 2 with const2 port 1 with a line
        dup1_port2 = comb_duplication.get_port_position(2)
        const2_port1 = comb_constructor2.get_port_position(1)
        line2 = Line(dup1_port2, const2_port1, color=TEXT)

        # connect dup2 port 2 const2 port 1 with a spline
        dup2_port2 = comb_duplication2.get_port_position(2)
        const2_port1 = comb_constructor2.get_port_position(2)
        spline1 = CubicBezier(
            dup2_port2,
            dup2_port2 + DOWN / 2,
            const2_port1 + UP / 2,
            const2_port1,
        ).set_stroke(TEXT, width=4)

        # connect dup1 port 1 with const 1 port 2 with a spline
        dup1_port1 = comb_duplication.get_port_position(1)
        const1_port2 = comb_constructor.get_port_position(1)
        spline2 = CubicBezier(
            dup1_port1,
            dup1_port1 + DOWN / 2,
            const1_port2 + UP / 2,
            const1_port2,
        ).set_stroke(TEXT, width=4)

        self.play(
            Create(line1),
            Create(line2),
            Create(spline1),
            Create(spline2),
        )

        # ===========
        self.next_slide()
        # ===========
        self.remove(
            dup_nodes,
            comb_duplication,
            comb_constructor,
            comb_duplication2,
            comb_constructor2,
            line1,
            line2,
            spline1,
            spline2,
        )

        # ===========
        self.next_slide()
        # ===========
        eras = CircleNode(port_len=0.6)
        constr = Node(nports=2)
        eras.rotate(PI)
        eras.move_to(UP)
        constr.move_to(DOWN * 0.6)
        self.add(eras, constr)
        self.wait()
        # ===========
        self.next_slide()
        # ===========
        eras2 = eras.copy()
        eras.generate_target().move_to(DOWN + LEFT)
        eras2.generate_target().move_to(DOWN + RIGHT)

        self.play(
            MoveToTarget(eras), MoveToTarget(eras2), FadeOut(constr, shift=UP)
        )

        # ===========
        self.next_slide()
        # ===========

        iter_text = Text("Iterative Duplication/Erasure").to_corner(UL)

        self.remove(eras, eras2)

        self.next_slide()
        law1 = Text(
            'Lafont: "The fundamental laws of computation', font_size=40
        )
        law2 = Text('are commutation and annihilation"', font_size=40).next_to(
            law1, DOWN, buff=0.5
        )

        self.play(FadeIn(law1), FadeIn(law2))
        self.next_slide()
        self.play(FadeOut(law1), FadeOut(law2))
        self.next_slide()

        self.play(Transform(comm_text, iter_text))

        comb_duplication.move_to(UP)
        comb_duplication.rotate(PI)
        package = Rectangle(
            width=2, height=2, stroke_width=4, stroke_color=TEXT
        )
        package.move_to(DOWN * 1.5)
        pport = Line(
            package.get_top(), package.get_top() + UP * 0.5, color=TEXT
        )

        # ===========
        self.next_slide()
        # ===========

        # label the ports of comb_duplication with x_1, x_2
        x_1 = MathTex("x_1", color=TEXT_SECONDARY, font_size=24)
        x_2 = MathTex("x_2", color=TEXT_SECONDARY, font_size=24)
        x_1.move_to(comb_duplication.get_port_position(2) + UP * 0.2)
        x_2.move_to(comb_duplication.get_port_position(1) + UP * 0.2)

        self.add(comb_duplication, package, pport, x_1, x_2)
        package2 = package.copy()
        pport2 = pport.copy()
        pack1 = VGroup(package, pport)
        pack2 = VGroup(package2, pport2)
        pack1.generate_target().move_to(ORIGIN + LEFT * 2)
        pack2.generate_target().move_to(ORIGIN + RIGHT * 2)
        comb_duplication.generate_target().move_to(DOWN).set_opacity(0)
        x_1.generate_target().move_to(LEFT * 2 + UP * 1.5)
        x_2.generate_target().move_to(RIGHT * 2 + UP * 1.5)

        self.wait()

        # ===========
        self.next_slide()
        # ===========
        self.play(
            MoveToTarget(pack1),
            MoveToTarget(pack2),
            MoveToTarget(comb_duplication),
            MoveToTarget(x_1),
            MoveToTarget(x_2),
        )

        # ===========
        self.next_slide()
        # ===========

        self.remove(pack1, pack2, x_1, x_2)

        eras = CircleNode(port_len=0.6)
        eras.rotate(PI)
        eras.move_to(UP)

        package = Rectangle(
            width=2, height=2, stroke_width=4, stroke_color=TEXT
        ).move_to(DOWN * 1.3)

        pport = package.get_top()
        pport_line = Line(pport, pport + UP * 0.5, color=TEXT)
        self.add(eras, package, pport_line)
        self.wait()
        # ===========
        self.next_slide()
        # ===========
        self.play(
            FadeOut(eras, shift=DOWN, runtime=1),
            FadeOut(VGroup(pport_line, package), shift=UP, runtime=1),
        )
        # ===========
        self.next_slide()
        # ===========

        title_loc = Text("Locality").to_corner(UL)
        self.play(Transform(comm_text, title_loc))

        a1 = Node(2)
        a2 = Node(2)
        a2.scale(0.5)
        a3 = CircleNode()
        a3.scale(0.5)
        a4 = Node(2, fill=TEXT)
        a4.rotate(PI)
        a4.move_to(UP + RIGHT)

        a5 = Node(2)
        a5.scale(0.5)
        a5.rotate(PI)

        a5.move_to(a4.get_port_position(1) - a5.get_port_position(0))

        a1.move_to(DOWN + RIGHT)

        a2.move_to(a1.get_port_position(1) - a2.get_port_position(0) + UP * 0.5)
        a3.move_to(a1.get_port_position(2) - a3.get_port_position() + UP * 0.3)

        self.add(a1, a2, a3, a4, a5)

        b1 = Node(2)
        b1.move_to(DOWN + LEFT)

        b4 = Node(2)
        b4.move_to(UP + LEFT)
        b4.rotate(PI)

        self.add(b1, b4)

        a5_p1 = a5.get_port_position(2)
        b4_p2 = b4.get_port_position(2)
        spline1 = CubicBezier(
            a5_p1, a5_p1 + UP, b4_p2 + UP * 2, b4_p2
        ).set_stroke(TEXT, width=4)

        a4_p1 = a4.get_port_position(2)
        b4_p1 = b4.get_port_position(1)
        spline2 = CubicBezier(a4_p1, a4_p1 + UP, b4_p1 + UP, b4_p1).set_stroke(
            TEXT, width=4
        )

        # now connect b1.1 with a2.1

        b1_p1 = b1.get_port_position(1)
        a2_p2 = a2.get_port_position(2)
        spline3 = CubicBezier(
            b1_p1, b1_p1 + DOWN * 2, a2_p2 + DOWN, a2_p2
        ).set_stroke(TEXT, width=4)

        # now connect b1.2 with a2.2
        b1_p2 = b1.get_port_position(2)
        a2_p1 = a2.get_port_position(1)
        spline4 = CubicBezier(
            b1_p2, b1_p2 + DOWN * 1.5, a2_p1 + DOWN, a2_p1
        ).set_stroke(TEXT, width=4)

        self.play(
            Create(spline1), Create(spline2), Create(spline3), Create(spline4)
        )

        self.wait()

        # ============
        self.next_slide()
        # ============
        # draw a box around b1, b2
        box = SurroundingRectangle(VGroup(b1, b4), color=TEXT, buff=0.1)
        box.set_stroke(TEXT_SECONDARY, width=4)
        self.play(Create(box))

        # ============
        self.next_slide()
        # ============
        # annihilate b1 and b4

        b1_p1 = b1.get_port_position(1)
        b1_p2 = b1.get_port_position(2)
        b4_p1 = b4.get_port_position(2)
        b4_p2 = b4.get_port_position(1)

        center = (b1_p1 + b1_p2 + b4_p1 + b4_p2) / 4
        tightness = 1

        # Control points to curve and cross the splines
        ctrl1a = (
            b1_p1 + UP * tightness + RIGHT * tightness
        )  # pull toward crossing
        ctrl1b = (
            b4_p2 + DOWN * tightness + LEFT * tightness
        )  # pull back toward target
        ctrl2a = (
            b1_p2 + UP * tightness + LEFT * tightness
        )  # opposite side curve
        ctrl2b = b4_p1 + DOWN * tightness + RIGHT * tightness

        # Create Bezier curves that cross
        curve1 = CubicBezier(b1_p1, ctrl1b, ctrl1a, b4_p2).set_stroke(TEXT, 4)
        curve2 = CubicBezier(b1_p2, ctrl2b, ctrl2a, b4_p1).set_stroke(TEXT, 4)

        self.play(
            FadeOut(b1, shift=UP, run_time=0.5),
            FadeOut(b4, shift=DOWN, run_time=0.5),
            FadeIn(curve1, run_time=1),
            FadeIn(curve2, run_time=1),
        )

        # ============
        self.next_slide()
        # ============
        self.remove(
            a1,
            a2,
            a3,
            a4,
            a5,
            spline1,
            spline2,
            spline3,
            spline4,
            curve1,
            curve2,
            box,
        )

        title_conflu = Text("One-Step Confluence").to_corner(UL)
        self.play(Transform(comm_text, title_conflu))

        red = MathTex(r"M\leadsto\dots\leadsto M'").next_to(
            code, DOWN, buff=0.2, aligned_edge=LEFT
        )

        spacing = 1.31415926

        M = MathTex("M")
        M.move_to(UP * spacing)

        M1 = MathTex("M_1")
        M1.move_to(LEFT * spacing)

        Mn = MathTex("M_n")
        Mn.move_to(LEFT * spacing + DOWN * spacing)

        M2 = MathTex("M_2")
        M2.move_to(RIGHT * spacing)

        Mm = MathTex("M_m")
        Mm.move_to(RIGHT * spacing + DOWN * spacing)

        M_PRIME = MathTex("M'")
        M_PRIME.move_to(DOWN * spacing * 2)

        # draw arrows from AC to BC and CC and from BC and CC to M_PRIME
        arrow1 = Arrow(
            M.get_bottom(),
            M1.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow2 = Arrow(
            M.get_bottom(),
            M2.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow3 = DashedLine(
            M1.get_bottom(),
            Mn.get_top(),
            color=TEXT,
            buff=0.1,
        )
        arrow4 = Arrow(
            Mn.get_bottom(),
            M_PRIME.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )
        arrow5 = DashedLine(
            M2.get_bottom(),
            Mm.get_top(),
            color=TEXT,
            buff=0.1,
        )
        arrow6 = Arrow(
            Mm.get_bottom(),
            M_PRIME.get_top(),
            color=TEXT,
            buff=0.1,
            max_tip_length_to_length_ratio=0.05,
        )

        # make the arrows thiner
        arrow1.set_stroke(width=2)
        arrow2.set_stroke(width=2)
        arrow3.set_stroke(width=2)
        arrow4.set_stroke(width=2)
        arrow5.set_stroke(width=2)
        arrow6.set_stroke(width=2)

        g = (
            VGroup(
                M,
                M1,
                Mn,
                M2,
                Mm,
                M_PRIME,
                arrow1,
                arrow2,
                arrow3,
                arrow4,
                arrow5,
                arrow6,
            )
            .move_to(ORIGIN)
            .shift([0, -1.5, 0])
        )

        weak = MathTex(r"n=m!").next_to(g, RIGHT, buff=0.5)
        opti = Text(r"always optimal??").next_to(weak, DOWN, buff=0.5)
        self.play(Write(red))
        self.next_slide()
        self.play(Create(g))
        self.wait()
        self.next_slide()
        self.play(g.animate.shift([-1.5, 0, 0]), Write(weak))
        self.next_slide()
        self.play(Write(opti))

        # BAD FOR PARALLELISM!

        # ============
        self.next_slide()
        # ============

        self.play(
            FadeOut(red),
            FadeOut(g),
            FadeOut(weak),
            FadeOut(opti),
        )

        # ============
        self.next_slide()
        # ============

        title_polar = Text("Polarization").to_corner(UL)

        self.play(Transform(comm_text, title_polar))

        node = Node(nports=2, round_radius=0.20)
        node.move_to(ORIGIN)

        # draw a "+" at port 0 and "-" at port 1 and 2 use MathText and TextSecondary as color
        plus = MathTex("\\oplus", font_size=32, color=TEXT_SECONDARY)
        minus = MathTex("\\ominus", font_size=32, color=TEXT_SECONDARY)
        plus.move_to(node.get_port_position(0) + UP * 0.2)
        minus1 = minus.copy()
        minus2 = minus.copy()
        minus1.move_to(node.get_port_position(1) + DOWN * 0.2)
        minus2.move_to(node.get_port_position(2) + DOWN * 0.2)
        self.play(Create(node), Write(plus), Write(minus1), Write(minus2))

        # ============
        self.next_slide()
        # ============

        node.generate_target().move_to(LEFT)
        plus.generate_target().move_to(
            node.get_port_position(0) + LEFT + UP * 0.2
        )
        minus1.generate_target().move_to(
            node.get_port_position(1) + LEFT + DOWN * 0.2
        )
        minus2.generate_target().move_to(
            node.get_port_position(2) + LEFT + DOWN * 0.2
        )

        node2 = Node(nports=2, round_radius=0.20)
        node2.move_to(RIGHT)

        plus2 = plus.copy()
        minus1_2 = minus1.copy()
        minus2_2 = minus2.copy()
        plus2.move_to(node2.get_port_position(0) + UP * 0.2)
        minus1_2.move_to(node2.get_port_position(1) + DOWN * 0.2)
        minus2_2.move_to(node2.get_port_position(2) + DOWN * 0.2)

        self.play(
            MoveToTarget(node),
            MoveToTarget(plus),
            MoveToTarget(minus1),
            MoveToTarget(minus2),
            Create(node2),
            Write(plus2),
            Write(minus1_2),
            Write(minus2_2),
        )

        # ============
        self.next_slide()
        # ============

        # connecjt both nodes at port 0 using a spline
        node0_pos = node.get_port_position(0)
        node2_0_pos = node2.get_port_position(0)
        # use a red indicatiing an error, dont use the Standard RED
        red = ManimColor("#FF0000")
        spline = CubicBezier(
            node0_pos, node0_pos + UP, node2_0_pos + UP, node2_0_pos
        ).set_stroke(red, width=4)

        self.play(Create(spline))

        # ============
        self.next_slide()
        # ============

        self.remove(spline)
        self.wait()
        # ============
        self.next_slide()
        # ============
        minus3 = minus.copy()
        minus3.move_to(node2.get_port_position(0) + UP * 0.2)
        self.play(Transform(plus2, minus3))

        # ============
        self.next_slide()
        # ============

        spline.set_stroke(TEXT, width=4)

        self.play(Create(spline))

        # ============
        self.next_slide()
        # ============
        self.remove(
            node, node2, plus2, minus1_2, minus2_2, plus, minus1, minus2, spline
        )

        pol_node = PolarizedNode(
            nports=2, round_radius=0.20, polarities=[True, False, False]
        )
        pol_node2 = PolarizedNode(
            nports=2, round_radius=0.20, polarities=[False, False, False]
        )

        pol_node.move_to(LEFT)
        pol_node2.move_to(RIGHT)

        self.add(pol_node, pol_node2)
        self.wait()

        # ============
        self.next_slide()
        # ============

        self.play(Create(spline))

        # ============
        self.next_slide()
        # ============

        # remove all nodes and spline
        self.remove(pol_node, pol_node2, spline)

        # TODO @Chris: port labels corresponding to LC!

        lambda_node = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[True, False, True],
            name="λ",
        )

        node2 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, False, False],
            fill=TEXT,
        )
        eras = PolarizedCircleNode(polarization=True)

        group1 = VGroup(lambda_node, node2, eras)
        desc1 = MathTex("(x \\Rightarrow M)")
        desc2 = MathTex(r"M(N)")

        at_node = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, True, False],
            name="@",
        )

        node2 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[True, True, True],
            fill=TEXT,
        )
        eras2 = PolarizedCircleNode(polarization=False)

        group2 = VGroup(at_node, node2, eras2)

        group1.arrange(RIGHT, aligned_edge=LEFT, buff=2)
        group2.arrange(RIGHT, aligned_edge=LEFT, buff=2)

        group3 = VGroup(group1, group2)
        group3.arrange(DOWN, buff=1)
        group3.move_to(ORIGIN)

        desc1.move_to(group1.get_y() + LEFT * 7)
        desc2.move_to(group2.get_y() + LEFT * 4)
        l_ret = MathTex(
            r"\text{ret}", font_size=24, color=TEXT_SECONDARY
        ).move_to(lambda_node.get_port_position(0) + UP * 0.2)
        l_x = MathTex("x", font_size=24, color=TEXT_SECONDARY).move_to(
            lambda_node.get_port_position(1) + LEFT * 0.3
        )
        l_m = MathTex("M", font_size=24, color=TEXT_SECONDARY).move_to(
            lambda_node.get_port_position(2) + RIGHT * 0.3
        )

        at_m = MathTex("M", font_size=24, color=TEXT_SECONDARY).move_to(
            at_node.get_port_position(0) + UP * 0.2
        )
        at_n = MathTex("N", font_size=24, color=TEXT_SECONDARY).move_to(
            at_node.get_port_position(1) + LEFT * 0.3
        )
        at_ret = MathTex(
            r"\text{ret}", font_size=24, color=TEXT_SECONDARY
        ).move_to(at_node.get_port_position(2) + RIGHT * 0.3)
        # TODO not properly alignend
        self.play(
            LaggedStartMap(FadeIn, group3, lag_ratio=0.1),
            FadeIn(desc1),
            FadeIn(desc2),
            Write(l_ret),
            Write(l_x),
            Write(l_m),
            Write(at_m),
            Write(at_n),
            Write(at_ret),
        )

        # ============
        self.next_slide()
        # ============

        betarule = MathTex(
            r"(x\Rightarrow M)(N)\leadsto M[x\mapsto N]"
        ).to_corner(UL)

        self.play(
            FadeIn(betarule),
            FadeOut(group1),
            FadeOut(group2),
            FadeOut(group3),
            FadeOut(desc1),
            FadeOut(desc2),
            FadeOut(comm_text),
            FadeOut(l_ret),
            FadeOut(l_x),
            FadeOut(l_m),
            FadeOut(at_m),
            FadeOut(at_n),
            FadeOut(at_ret),
        )

        node_at = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, True, False],
            name="@",
        )
        node_at.move_to(UP)
        node_at.rotate(PI)
        node_at_port_labels = [
            MathTex("M", font_size=24, color=TEXT_SECONDARY).move_to(
                node_at.get_port_position(0) + RIGHT * 0.3
            ),
            MathTex("N", font_size=24, color=TEXT_SECONDARY).move_to(
                node_at.get_port_position(1) + RIGHT * 0.3
            ),
            MathTex(r"\text{ret}", font_size=24, color=TEXT_SECONDARY).move_to(
                node_at.get_port_position(2) + LEFT * 0.3
            ),
        ]

        node_lambda = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[True, False, True],
            name="λ",
        )
        node_lambda.move_to(DOWN * 1.1)
        node_lambda_port_labels = [
            MathTex("x", font_size=24, color=TEXT_SECONDARY).move_to(
                node_lambda.get_port_position(1) + LEFT * 0.3
            ),
            MathTex("M", font_size=24, color=TEXT_SECONDARY).move_to(
                node_lambda.get_port_position(2) + RIGHT * 0.3
            ),
        ]
        self.play(
            Create(node_at),
            *[Write(label) for label in node_at_port_labels],
            Create(node_lambda),
            *[Write(label) for label in node_lambda_port_labels],
        )

        # ============
        self.next_slide()
        # ============
        p1 = node_lambda.get_port_position(1)
        p2 = node_lambda.get_port_position(2)
        p3 = node_at.get_port_position(1)
        p4 = node_at.get_port_position(2)

        center = (p1 + p2 + p3 + p4) / 4
        tightness = 1

        # Control points to curve and cross the splines
        ctrl1a = p1 + UP * tightness + RIGHT * tightness  # pull toward crossing
        ctrl1b = (
            p3 + DOWN * tightness + LEFT * tightness
        )  # pull back toward target

        ctrl2a = p2 + UP * tightness + LEFT * tightness  # opposite side curve
        ctrl2b = p4 + DOWN * tightness + RIGHT * tightness

        # Create Bezier curves that cross
        curve1 = CubicBezier(p1, ctrl1b, ctrl1a, p3).set_stroke(TEXT, 4)
        curve2 = CubicBezier(p2, ctrl2b, ctrl2a, p4).set_stroke(TEXT, 4)

        arrow1 = Arrow(
            p1 + UP * 0.5 + LEFT * 0.005,
            p1 + DOWN * 0.2 + LEFT * 0.005,
            color=TEXT,
            stroke_width=2,
            max_tip_length_to_length_ratio=3,
        )

        arrow2 = Arrow(
            p4 + LEFT * 0.008,
            p4 + DOWN * 0.7 + LEFT * 0.008,
            color=TEXT,
            stroke_width=2,
            max_tip_length_to_length_ratio=3,
        ).rotate(PI)

        self.play(
            FadeOut(node_at, shift=DOWN, run_time=0.5),
            FadeOut(node_lambda, shift=UP, run_time=0.5),
            FadeIn(curve1, run_time=1),
            FadeIn(curve2, run_time=1),
            FadeIn(arrow1, run_time=1.5),
            FadeIn(arrow2, run_time=1.5),
        )

        # ============
        self.next_slide()
        # ============
        self.remove(
            node_at,
            node_lambda,
            *node_at_port_labels,
            *node_lambda_port_labels,
            curve1,
            curve2,
            arrow1,
            arrow2,
        )
        self.play(FadeOut(betarule))

        title_formula = MathTex(
            "(f \\Rightarrow f(f)) (x \\Rightarrow x)"
        ).to_corner(UL)

        cite_enrico = (
            Text("Example by Enrico Z. Borba").scale(0.3).to_corner(DL)
        )

        node1 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, False, False],
            fill=TEXT,
        )
        node_at1 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, True, False],
            name="@",
        )
        node_at2 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[False, True, False],
            name="@",
        )
        node_lambda1 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[True, False, True],
            name="λ",
        )
        node_lambda2 = PolarizedNode(
            nports=2,
            round_radius=0.20,
            polarities=[True, False, True],
            name="λ",
        )

        node1.move_to(LEFT * 2)
        node_at1.move_to(LEFT + DOWN * 2.5)
        node_lambda1.move_to(LEFT + UP * 2.5)
        node_at2.move_to(UP * 2.5 + RIGHT)
        node_lambda2.move_to(RIGHT * 2)

        # connect node1 port1 with node_at1 port1 with a spline
        node1_p1 = node1.get_port_position(1)
        node_at1_p1 = node_at1.get_port_position(1)
        spline1 = CubicBezier(
            node1_p1, node1_p1 + DOWN * 2.5, node_at1_p1 + DOWN, node_at1_p1
        ).set_stroke(TEXT, width=4)

        # connect node1 port2 with node_at1 port0 with a spline

        node1_p2 = node1.get_port_position(2)
        node_at1_p0 = node_at1.get_port_position(0)

        spline2 = CubicBezier(
            node1_p2, node1_p2 + DOWN * 0.4, node_at1_p0 + UP * 0.4, node_at1_p0
        ).set_stroke(TEXT, width=4)

        # connect node1 port0 with node_lambda1 port1 with a spline

        node1_p0 = node1.get_port_position(0)
        node_lambda1_p1 = node_lambda1.get_port_position(1)
        spline3 = CubicBezier(
            node1_p0,
            node1_p0 + UP * 0.4,
            node_lambda1_p1 + DOWN * 0.4,
            node_lambda1_p1,
        ).set_stroke(TEXT, width=4)

        # connect node_at1 port2 with node_lambda1 port2 with a spline
        node_at1_p2 = node_at1.get_port_position(2)
        node_lambda1_p2 = node_lambda1.get_port_position(2)
        spline4 = CubicBezier(
            node_at1_p2,
            node_at1_p2 + DOWN * 2 + RIGHT,
            node_lambda1_p2 + DOWN * 0.4,
            node_lambda1_p2,
        ).set_stroke(TEXT, width=4)

        # connect node_lambda1 port0 with node_at_2 port0 with a spline
        node_lambda1_p0 = node_lambda1.get_port_position(0)
        node_at2_p0 = node_at2.get_port_position(0)
        spline5 = CubicBezier(
            node_lambda1_p0,
            node_lambda1_p0 + UP * 0.4,
            node_at2_p0 + UP * 0.4,
            node_at2_p0,
        ).set_stroke(TEXT, width=4)

        # connect node_a2 port1 with lambda1 port0 with a spline
        node_at2_p1 = node_at2.get_port_position(1)
        node_lambda2_p0 = node_lambda2.get_port_position(0)
        spline6 = CubicBezier(
            node_at2_p1,
            node_at2_p1 + DOWN * 0.4,
            node_lambda2_p0 + UP * 0.4,
            node_lambda2_p0,
        ).set_stroke(TEXT, width=4)

        # connect node_lamdba2 port1 with lambda1 port2 with a spline

        node_lambda2_p1 = node_lambda2.get_port_position(1)
        node_lambda2_p2 = node_lambda2.get_port_position(2)
        spline7 = CubicBezier(
            node_lambda2_p1,
            node_lambda2_p1 + DOWN * 0.4,
            node_lambda2_p2 + DOWN * 0.4,
            node_lambda2_p2,
        ).set_stroke(TEXT, width=4)

        self.play(
            Create(node1),
            Create(node_at1),
            Create(node_lambda1),
            Create(node_at2),
            Create(node_lambda2),
            Create(spline1),
            Create(spline2),
            Create(spline3),
            Create(spline4),
            Create(spline5),
            Create(spline6),
            Create(spline7),
            Write(title_formula),
            Write(cite_enrico),
        )

        # ============
        self.next_slide()
        # ============
        self.remove(
            node1,
            node_at1,
            node_lambda1,
            node_at2,
            node_lambda2,
            spline1,
            spline2,
            spline3,
            spline4,
            spline5,
            spline6,
            spline7,
            title_formula,
            cite_enrico,
            title_polar,
            title_loc,
        )

        # ============
        self.next_slide()
        # ============

        title_recap = Text("Recap", color=TEXT).to_corner(UL)

        blist = BulletedList(
            "Only 3 agents",
            "Turing complete",
            "Elegant \\& explicit memory management",
            "Massively parallel",
            "LC encoding (with readback)",
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Write(title_recap),
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )

        # ==============
        self.next_slide()
        # ==============
        catch = Text(
            "Is there a Catch???",
            font_size=80,
            weight=BOLD,
            color=TEXT_SECONDARY,
        )
        self.add(catch)
        self.wait()

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(blist), FadeOut(catch))

        title_bookkeeping = Text("Bookkeeping", color=TEXT).to_corner(UL)
        self.play(Transform(title_recap, title_bookkeeping))

        comb_duplication = Node(nports=2, round_radius=0.20, fill=TEXT)

        comb_constructor = Node(nports=2, round_radius=0.20, fill=TEXT)

        dup_nodes = connect_nodes(comb_constructor, comb_duplication, 0, 0)
        dup_nodes.move_to(ORIGIN)
        self.add(dup_nodes)
        self.wait()

        # ===========
        self.next_slide()
        # ===========

        comb_duplication2 = comb_duplication.copy()
        comb_constructor2 = comb_constructor.copy()

        dist = 1.5

        comb_duplication.generate_target().move_to(UP * dist + RIGHT)
        comb_duplication2.generate_target().move_to(UP * dist + LEFT)
        comb_constructor.generate_target().move_to(DOWN * dist + LEFT)
        comb_constructor2.generate_target().move_to(DOWN * dist + RIGHT)
        self.play(
            MoveToTarget(comb_constructor),
            MoveToTarget(comb_duplication),
            MoveToTarget(comb_constructor2),
            MoveToTarget(comb_duplication2),
        )

        # connect dup2 port 1 with const1 port 2 with a line
        dup2_port1 = comb_duplication2.get_port_position(1)
        const1_port2 = comb_constructor.get_port_position(2)
        line1 = Line(dup2_port1, const1_port2, color=TEXT)

        # connect dup1 port 2 with const2 port 1 with a line
        dup1_port2 = comb_duplication.get_port_position(2)
        const2_port1 = comb_constructor2.get_port_position(1)
        line2 = Line(dup1_port2, const2_port1, color=TEXT)

        # connect dup2 port 2 const2 port 1 with a spline
        dup2_port2 = comb_duplication2.get_port_position(2)
        const2_port1 = comb_constructor2.get_port_position(2)
        spline1 = CubicBezier(
            dup2_port2,
            dup2_port2 + DOWN / 2,
            const2_port1 + UP / 2,
            const2_port1,
        ).set_stroke(TEXT, width=4)

        # connect dup1 port 1 with const 1 port 2 with a spline
        dup1_port1 = comb_duplication.get_port_position(1)
        const1_port2 = comb_constructor.get_port_position(1)
        spline2 = CubicBezier(
            dup1_port1,
            dup1_port1 + DOWN / 2,
            const1_port2 + UP / 2,
            const1_port2,
        ).set_stroke(TEXT, width=4)

        self.play(
            Create(line1),
            Create(line2),
            Create(spline1),
            Create(spline2),
        )

        # ===========
        self.next_slide()
        # ===========
        self.remove(
            dup_nodes,
            comb_duplication,
            comb_constructor,
            comb_duplication2,
            comb_constructor2,
            line1,
            line2,
            spline1,
            spline2,
        )

        blist = BulletedList(
            "Abstract: Terms that duplicate arguments can not be duplicated",
            "Labels: If terms that duplicate their arguments are duplicated, the duplicates can not duplicate eachother",
            "Bookkeeping: No limitations, full lambda calculus",
            width=300,
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(blist))

        title_optimality = Text("LC Optimality", color=TEXT).to_corner(UL)

        blist = BulletedList(
            "Shortest reduction path",
            "Oracle overhead?",
            "Detached Nets?",
            "Parallelism?",
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Transform(title_recap, title_optimality),
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(blist))

        title_extensions = Text("Extension Agents", color=TEXT).to_corner(UL)

        blist = BulletedList(
            "Oracle (for optimality/bookkeeping)",
            "Reference agents",
            "Primitive/builtin operators",
            "Ambiguous agents (parallel OR)",
            "Action agents (IO)",
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Transform(title_recap, title_extensions),
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(blist))

        title_usecases = Text("Usecases", color=TEXT).to_corner(UL)

        blist = BulletedList(
            "Massive parallelism",
            "Optimal reduction",
            "A way of thinking about computation",
            "Tool for arguing about reduction order etc.",
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Transform(title_recap, title_usecases),
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )

        # ============
        self.next_slide()
        # ============

        self.play(FadeOut(blist))

        title_resources = Text("Resources", color=TEXT).to_corner(UL)

        blist = BulletedList(
            "General: github/marvinborner/interaction-net-resources",
            "IO: github/marvinborner/optimal-effects",
            "Slides: github/anymelfarm/gpn23-slides",
            "Or just ask us about anything! (DECT: 5063)",
        )
        blist.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(
            Transform(title_recap, title_resources),
            LaggedStartMap(FadeIn, blist, shift=UP, lag_ratio=0.1),
        )
