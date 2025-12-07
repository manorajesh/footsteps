use std::f64;

#[derive(Debug, Copy, Clone)]
struct Vector2 {
    x: f64,
    y: f64,
}

impl Vector2 {
    fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(&self) -> Vector2 {
        let mag = self.magnitude();
        if mag == 0.0 {
            Vector2 { x: 0.0, y: 0.0 }
        } else {
            Vector2 { x: self.x / mag, y: self.y / mag }
        }
    }
}

fn main() {
    let prev = Vector2 { x: 100.0, y: 80.0 };
    let curr = Vector2 { x: 112.0, y: 95.0 };

    let orig_vector = Vector2 { x: curr.x - prev.x, y: curr.y - prev.y };
    let unit_vector = orig_vector.normalize();

    println!("Unit vector: ({}, {})", unit_vector.x, unit_vector.y);
}
