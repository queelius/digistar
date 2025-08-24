; Solar System creation script
; Creates a simplified solar system with planets orbiting the sun

; Create the Sun
(define sun (star :mass 1.989e30 :temp 5778 :pos [0 0]))

; Helper function to create a planet in orbit
(define make-planet (lambda (name semi-major ecc mass)
  (particle 
    :mass mass
    :pos [(* semi-major 1.496e11) 0]  ; Convert AU to meters
    :vel [0 (sqrt (/ (* gravity-constant sun-mass) 
                     (* semi-major 1.496e11)))])))

; Create inner planets
(define mercury (make-planet "Mercury" 0.387 0.206 3.301e23))
(define venus   (make-planet "Venus"   0.723 0.007 4.867e24))
(define earth   (make-planet "Earth"   1.000 0.017 5.972e24))
(define mars    (make-planet "Mars"    1.524 0.093 6.417e23))

; Create asteroid belt as a cloud
(cloud 
  :n 1000
  :center [0 0]
  :radius (* 2.7 1.496e11)  ; ~2.7 AU
  :mass-min 1e15
  :mass-max 1e20
  :temp 150)

; Create gas giants
(define jupiter (make-planet "Jupiter" 5.203 0.048 1.898e27))
(define saturn  (make-planet "Saturn"  9.537 0.054 5.683e26))

; Add some moons to Earth
(let ([earth-x (measure 'position earth)]
      [moon-dist 3.844e8])
  (particle
    :mass 7.342e22
    :pos [(+ (first earth-x) moon-dist) (second earth-x)]
    :vel [0 1022]))  ; Orbital velocity of moon

; Create a persistent rule for comet behavior
(rule 'comet-tail
  (when (< (distance particle sun) (* 2 1.496e11))  ; Within 2 AU
    (explode particle :energy 1e10 :fragments 5)))

; Monitor system energy
(watch
  :every 1.0
  (measure 'total-energy (find :radius 1e13)))